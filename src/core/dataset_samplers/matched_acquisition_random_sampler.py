"""Random control that matches another policy's teacher-trace acquisition curve."""

from typing import override

import pandas as pd
from pydantic import model_validator

from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.dataset_samplers.matched_acquisition import AcquisitionStep, epoch_selection, uniform_key


class MatchedAcquisitionRandomSamplerConfig(BaseDatasetSamplerConfig):
    # Per-epoch (new, total) counts to reproduce, e.g. acquisition_schedule() of an entropy-gain
    # top-k run's selections. Required: the control is only meaningful against a recorded curve.
    schedule: list[AcquisitionStep]
    seed: int = 42

    @model_validator(mode="after")
    def _validate(self):
        # Unselected rows score 0; they must be dropped rather than padded back in up to top_k.
        if not self.drop_non_positive:
            raise ValueError("MatchedAcquisitionRandomSampler requires drop_non_positive=True")
        if not self.schedule:
            raise ValueError("schedule must cover at least one epoch")
        return self


class MatchedAcquisitionRandomSampler(BaseDatasetSampler):
    """Trains on the questions epoch_selection draws for the current resampling epoch.

    Selected rows get a uniform (seed, epoch, question)-keyed score and all others score 0, so
    top_k picks a uniformly random subset of the epoch's selection. Two samplers with the same
    schedule and seed but different top_k (the 1024 trace / 256 single-token mix) therefore get
    nested selections, like RandomSampler's shared random_value column.
    """

    config: MatchedAcquisitionRandomSamplerConfig

    def __init__(self, config: MatchedAcquisitionRandomSamplerConfig):
        super().__init__(config)
        self._scores: dict[str, float] = {}

    @override
    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        if "question_id" not in df.columns:
            raise KeyError("MatchedAcquisitionRandomSampler requires a question_id column")
        selected = epoch_selection(df["question_id"].astype(str), self.config.schedule, self.epoch, self.config.seed)
        self._scores = {q: uniform_key(self.config.seed, "order", self.epoch, q) for q in selected}
        return super()._select(df)

    @override
    def _score_row(self, row: dict) -> float:
        return self._scores.get(str(row["question_id"]), 0.0)
