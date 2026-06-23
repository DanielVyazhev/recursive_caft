"""Supervise a GPU work unit in a fresh subprocess, restarting it on crash.

The eval/estimation GPU is flaky and dies with a native SIGSEGV (EXIT=139) mid-run.
Running each unit in its own `sys.executable` subprocess (not a fork) gives a full
resource reset (CUDA context, caching allocator, host RAM) even on clean exit and
isolates crashes — the parent does no GPU work, so it survives. `supervise_unit`
restarts a crashed worker with exponential backoff, bounded by a circuit breaker that
detects deterministic failures (repeated fast crashes) so we never spin forever.

Resume rides whatever per-unit checkpoints the worker writes; the supervisor just
re-runs the same command.
"""

import os
import signal
import subprocess
import time

from core.utils.logger import logger


def terminate_process_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """SIGTERM the worker's whole process group, then SIGKILL stragglers.

    The worker is spawned with start_new_session=True, so it leads its own process
    group; signalling that group reaches the worker *and* any descendants it spawned
    (DataLoader / vLLM workers) — which proc.send_signal() to the direct child would
    miss. SIGTERM first lets runtime_trace's handler shut down cleanly and flush
    logs; SIGKILL after grace_s guarantees teardown if it's wedged in a CUDA C call.
    Catching KeyboardInterrupt means an impatient second Ctrl+C escalates immediately.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            logger.warning(f"[unit] worker pid={proc.pid} ignored SIGTERM after {grace_s:.0f}s — sending SIGKILL")
    except KeyboardInterrupt:
        logger.warning("[unit] second interrupt — sending SIGKILL now")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def supervise_unit(
    cmd: list[str],
    env: dict[str, str],
    *,
    label: str,
    min_healthy_s: float,
    max_fast: int,
    max_attempts: int,
) -> None:
    """Run `cmd` in a fresh subprocess, restarting on any nonzero exit until it exits 0.

    The flaky GPU dies with SIGSEGV/139; resume rides the worker's own checkpoints. A
    circuit breaker hard-stops on repeated fast failures (a deterministic bug, not
    transient infra) so we never spin forever.

    Interrupts (Ctrl+C / scheduler SIGTERM) must stop the unit, not be mistaken for a
    transient crash and retried. The worker is spawned in its own session
    (start_new_session=True) so terminal Ctrl+C reaches only this supervisor; we then
    tear down the worker's whole process group and re-raise, breaking the retry loop
    and propagating out so the program exits.

    `label` identifies the unit in logs (e.g. "dataset=3" or "epoch=1").
    """
    consecutive_fast = 0
    for attempt in range(1, max_attempts + 1):
        start = time.monotonic()
        logger.info(f"[unit] {label} starting worker attempt={attempt}")
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            code = proc.wait()
        except BaseException as exc:  # KeyboardInterrupt (Ctrl+C) or SystemExit (parent SIGTERM)
            logger.warning(
                f"[unit] {label} supervisor interrupted by {type(exc).__name__} — "
                f"terminating worker group pid={proc.pid}"
            )
            terminate_process_group(proc)
            raise
        dur = time.monotonic() - start
        if code == 0:
            logger.info(f"[unit] {label} worker attempt={attempt} finished in {dur:.0f}s.")
            return
        consecutive_fast = consecutive_fast + 1 if dur < min_healthy_s else 0
        logger.warning(
            f"[unit] {label} worker attempt={attempt} pid={proc.pid} crashed: "
            f"exit={code} after {dur:.0f}s (consecutive_fast={consecutive_fast})"
        )
        if consecutive_fast >= max_fast:
            raise RuntimeError(
                f"[unit] {label}: {consecutive_fast} consecutive crashes in "
                f"<{min_healthy_s:.0f}s — looks deterministic, not transient infra. "
                f"Aborting (last exit={code})."
            )
        backoff = min(30.0, 2.0**consecutive_fast)
        logger.info(f"[unit] {label} restarting in {backoff:.0f}s …")
        time.sleep(backoff)
    raise RuntimeError(f"[unit] {label}: exceeded max_attempts={max_attempts}.")
