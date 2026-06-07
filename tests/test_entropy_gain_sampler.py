"""Sampling semantics: entropy-gain scoring and top-k selection (pure pandas)."""

import pandas as pd
import pytest
from datasets import Dataset

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_sampler import EntropyGainSampler


def test_score_is_clamped_gain():
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=1))
    assert sampler._score_row({"entropy_value": 0.9, "teacher_entropy": 0.2}) == pytest.approx(0.7)
    # Negative gain clamps to 0.
    assert sampler._score_row({"entropy_value": 0.1, "teacher_entropy": 0.5}) == 0.0


def test_create_sample_selects_top_k_by_gain():
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "entropy_value": [1.0, 0.5, 0.9, 0.2],
            "teacher_entropy": [0.1, 0.1, 0.1, 0.1],
        }
    )  # gains: a=0.9, b=0.4, c=0.8, d=0.1
    sampler = EntropyGainSampler(BaseDatasetSamplerConfig(top_k=2))

    result = sampler.create_sample(Dataset.from_pandas(df))

    assert len(result) == 2
    assert set(result["id"]) == {"a", "c"}
