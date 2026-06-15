"""supervise_unit: restart-on-crash with a circuit breaker for deterministic failures.

CPU-only — drives the loop with tiny python subprocesses (and a fake Popen for the
interrupt case). The process-group teardown itself is covered by
test_eval_subprocess_interrupt.py.
"""

import sys

import pytest

import core.utils.subprocess_supervision as sup
from core.utils.subprocess_supervision import supervise_unit

# Each appends one line to argv[1] (an attempt counter), then exits with the given code.
_CRASH = "import sys\nwith open(sys.argv[1], 'a') as f: f.write('x\\n')\nsys.exit(1)\n"
_OK = "import sys\nwith open(sys.argv[1], 'a') as f: f.write('x\\n')\nsys.exit(0)\n"


def _cmd(script, counter):
    return [sys.executable, "-c", script, str(counter)]


def _attempts(counter):
    return len(counter.read_text().split()) if counter.exists() else 0


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    # Don't actually sleep between restart attempts.
    monkeypatch.setattr(sup.time, "sleep", lambda _s: None)


def test_exit_zero_returns_after_one_attempt(tmp_path):
    counter = tmp_path / "n.txt"
    supervise_unit(
        _cmd(_OK, counter), {}, label="ok", min_healthy_s=0.0, max_fast=3, max_attempts=5
    )
    assert _attempts(counter) == 1


def test_fast_crashes_trip_circuit_breaker(tmp_path):
    counter = tmp_path / "n.txt"
    # min_healthy_s huge → every immediate crash counts as "fast" → abort at max_fast.
    with pytest.raises(RuntimeError, match="consecutive crashes"):
        supervise_unit(
            _cmd(_CRASH, counter), {}, label="cb", min_healthy_s=1e9, max_fast=3, max_attempts=50
        )
    assert _attempts(counter) == 3


def test_non_fast_crashes_run_to_max_attempts(tmp_path):
    counter = tmp_path / "n.txt"
    # min_healthy_s=0 → no crash is "fast" → consecutive_fast never accumulates → max_attempts hit.
    with pytest.raises(RuntimeError, match="exceeded max_attempts"):
        supervise_unit(
            _cmd(_CRASH, counter), {}, label="ma", min_healthy_s=0.0, max_fast=3, max_attempts=4
        )
    assert _attempts(counter) == 4


def test_interrupt_terminates_group_and_reraises(monkeypatch):
    # proc.wait() raising KeyboardInterrupt (Ctrl+C) must tear down the group and propagate,
    # not be retried as a transient crash.
    terminated = []

    class _FakePopen:
        def __init__(self, *a, **k):
            self.pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(sup.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(sup, "terminate_process_group", lambda proc, **k: terminated.append(proc.pid))

    with pytest.raises(KeyboardInterrupt):
        supervise_unit(["x"], {}, label="int", min_healthy_s=0.0, max_fast=3, max_attempts=5)

    assert terminated == [4242]


def test_passes_env_and_session_to_popen(monkeypatch):
    seen = {}

    class _FakePopen:
        def __init__(self, cmd, env=None, start_new_session=None):
            seen["cmd"], seen["env"], seen["sns"] = cmd, env, start_new_session
            self.pid = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(sup.subprocess, "Popen", _FakePopen)
    supervise_unit(["a", "b"], {"K": "V"}, label="env", min_healthy_s=0.0, max_fast=3, max_attempts=2)
    assert seen == {"cmd": ["a", "b"], "env": {"K": "V"}, "sns": True}
