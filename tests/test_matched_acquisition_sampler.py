import pandas as pd
import pytest
from datasets import Dataset
from pydantic import ValidationError

from core.dataset_samplers.matched_acquisition import AcquisitionStep, acquisition_schedule, epoch_selection
from core.dataset_samplers.matched_acquisition_random_sampler import (
    MatchedAcquisitionRandomSampler,
    MatchedAcquisitionRandomSamplerConfig,
)

POOL = [f"q{i}" for i in range(100)]
# Epoch 0 acquires 20; later epochs acquire progressively fewer and reuse the rest.
SCHEDULE = [AcquisitionStep(20, 20), AcquisitionStep(10, 20), AcquisitionStep(5, 20), AcquisitionStep(0, 20)]


def _sampler(top_k: int = 20, seed: int = 42, schedule=SCHEDULE) -> MatchedAcquisitionRandomSampler:
    return MatchedAcquisitionRandomSampler(
        MatchedAcquisitionRandomSamplerConfig(top_k=top_k, schedule=schedule, seed=seed)
    )


def _ds() -> Dataset:
    return Dataset.from_pandas(pd.DataFrame({"question_id": POOL, "entropy_value": [0.5] * len(POOL)}))


def test_acquisition_schedule_counts_new_and_total():
    selections = [["a", "b", "c"], ["b", "c", "d"], ["a", "d"], ["e", "a"]]
    assert acquisition_schedule(selections) == [(3, 3), (1, 3), (0, 2), (1, 2)]


def test_selection_replays_the_curve():
    acquired: set[str] = set()
    for epoch, step in enumerate(SCHEDULE):
        selected = epoch_selection(POOL, SCHEDULE, epoch, seed=42)
        assert len(selected) == step.total
        assert len(selected - acquired) == step.new
        acquired |= selected
    assert len(acquired) == sum(step.new for step in SCHEDULE)
    # Replaying the control's own selections recovers the schedule it was given.
    selections = [epoch_selection(POOL, SCHEDULE, epoch, seed=42) for epoch in range(len(SCHEDULE))]
    assert acquisition_schedule(selections) == SCHEDULE


def test_reuse_is_redrawn_each_epoch():
    # Epoch 3 acquires nothing new: all of it is reuse, and it must not simply repeat epoch 2.
    assert epoch_selection(POOL, SCHEDULE, 3, seed=42) != epoch_selection(POOL, SCHEDULE, 2, seed=42)


def test_selection_ignores_pool_order_and_depends_on_seed():
    shuffled = list(reversed(POOL))
    assert epoch_selection(POOL, SCHEDULE, 1, seed=42) == epoch_selection(shuffled, SCHEDULE, 1, seed=42)
    assert epoch_selection(POOL, SCHEDULE, 1, seed=42) != epoch_selection(POOL, SCHEDULE, 1, seed=43)


@pytest.mark.parametrize(
    ("schedule", "epoch"),
    [
        (SCHEDULE, 4),  # beyond the schedule
        ([AcquisitionStep(10, 20)], 0),  # reuses more than was acquired before
        ([AcquisitionStep(80, 80), AcquisitionStep(30, 30)], 1),  # acquires more than the pool
        ([AcquisitionStep(20, 10)], 0),  # new > total
    ],
)
def test_invalid_schedule_raises(schedule, epoch):
    with pytest.raises(ValueError):
        epoch_selection(POOL, schedule, epoch, seed=42)


def test_sampler_selects_the_epoch_selection():
    sampler = _sampler()
    for epoch in range(len(SCHEDULE)):
        sampler.set_epoch(epoch)
        ids = set(sampler.create_sample(_ds())["question_id"])
        assert ids == epoch_selection(POOL, SCHEDULE, epoch, seed=42)


def test_smaller_top_k_is_a_nested_subset():
    trace, single_token = _sampler(top_k=20), _sampler(top_k=5)
    trace.set_epoch(2)
    single_token.set_epoch(2)

    trace_ids = set(trace.create_sample(_ds())["question_id"])
    single_token_ids = set(single_token.create_sample(_ds())["question_id"])

    assert len(single_token_ids) == 5
    assert single_token_ids < trace_ids


def test_count_selected_matches_create_sample_and_leaks_no_column():
    sampler = _sampler()
    sampler.set_epoch(1)
    df = pd.DataFrame({"question_id": POOL})
    assert sampler.count_selected(df) == SCHEDULE[1].total
    assert "score" not in df.columns


def test_config_requires_drop_non_positive_and_a_schedule():
    with pytest.raises(ValidationError):
        MatchedAcquisitionRandomSamplerConfig(top_k=5, schedule=SCHEDULE, drop_non_positive=False)
    with pytest.raises(ValidationError):
        MatchedAcquisitionRandomSamplerConfig(top_k=5, schedule=[])


def test_schedule_round_trips_through_json_lists():
    config = MatchedAcquisitionRandomSamplerConfig(top_k=5, schedule=[[20, 20], [10, 20]])
    assert config.schedule == [AcquisitionStep(20, 20), AcquisitionStep(10, 20)]
