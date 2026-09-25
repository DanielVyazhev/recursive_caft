"""Selection logic for the matched-acquisition random control (stdlib only).

A policy that re-selects its training sample every epoch "acquires" a question -- needs its teacher
trace -- the first time it selects it, and reuses the trace on later selections. Its acquisition
curve is the per-epoch pair (new, total): questions selected for the first time, and questions
selected at all. The matched control reproduces a given curve with uniformly random questions, so
it spends exactly the same teacher traces and student compute per epoch as the policy it mirrors;
only *which* questions are acquired and reused differs.

MatchedAcquisitionRandomSampler adapts epoch_selection to the BaseDatasetSampler interface.
"""

from collections.abc import Iterable, Sequence
from hashlib import blake2b
from typing import NamedTuple


class AcquisitionStep(NamedTuple):
    # Questions selected this epoch that no earlier epoch selected (new teacher traces).
    new: int
    # Questions selected this epoch in total (new + reused).
    total: int


def acquisition_schedule(selections: Sequence[Iterable[str]]) -> list[AcquisitionStep]:
    """The acquisition curve of a sequence of per-epoch selections (question ids, epoch 0 first)."""
    seen: set[str] = set()
    schedule = []
    for selection in selections:
        selected = set(selection)
        schedule.append(AcquisitionStep(new=len(selected - seen), total=len(selected)))
        seen |= selected
    return schedule


def uniform_key(*parts: object) -> float:
    """A reproducible uniform draw in (0, 1], keyed by its parts (no RNG state, resume-safe)."""
    payload = ":".join(str(part) for part in parts).encode()
    integer = int.from_bytes(blake2b(payload, digest_size=8).digest(), "big")
    return (integer + 1) / 2**64


def epoch_selection(
    question_ids: Iterable[str], schedule: Sequence[AcquisitionStep], epoch: int, seed: int
) -> set[str]:
    """Questions the control trains on at `epoch`.

    The pool is put in a fixed, seed-keyed random acquisition order. Epoch e acquires the next
    schedule[e].new questions in that order, and fills the remaining schedule[e].total - new slots
    with a uniform draw (fresh each epoch) from the questions acquired before e. Keying on the
    question id rather than its row position keeps the order stable if a dump reorders rows.
    """
    if not 0 <= epoch < len(schedule):
        raise ValueError(f"No acquisition step for epoch {epoch}; the schedule covers {len(schedule)} epochs")

    pool = sorted(set(question_ids), key=lambda q: uniform_key(seed, "acquire", q))
    acquired_before = sum(step.new for step in schedule[:epoch])
    step = schedule[epoch]
    reused = step.total - step.new

    if step.new < 0 or reused < 0:
        raise ValueError(f"Invalid acquisition step {step} at epoch {epoch}")
    if acquired_before + step.new > len(pool):
        raise ValueError(
            f"Epoch {epoch} needs {acquired_before + step.new} acquired questions but the pool has {len(pool)}"
        )
    if reused > acquired_before:
        raise ValueError(f"Epoch {epoch} reuses {reused} questions but only {acquired_before} were acquired before it")

    new = pool[acquired_before : acquired_before + step.new]
    reuse = sorted(pool[:acquired_before], key=lambda q: uniform_key(seed, "reuse", epoch, q))[:reused]
    return set(new) | set(reuse)
