from abc import ABC, abstractmethod
from typing import cast

import pandas as pd
from datasets import Dataset
from pydraconf import PydraConfig


class BaseDatasetSamplerConfig(PydraConfig):
    top_k: int
    # Discard rows whose score is <= 0 before selecting top_k. A non-positive score means the row
    # carries no learning signal (e.g. for entropy gain, the student is already at or below the
    # teacher's entropy), so it is dropped — even if that leaves fewer than top_k rows. This also
    # drops NaN-score rows (NaN > 0 is False), e.g. rows whose entropy could not be measured.
    drop_non_positive: bool = True
    # Shuffle the selected rows instead of presenting them in easy-to-hard score order. Selection
    # membership is unchanged. Each create_sample call uses shuffle_seed + the call index, giving
    # successive epochs different permutations while keeping a fresh run reproducible.
    shuffle: bool = False
    shuffle_seed: int = 42


class BaseDatasetSampler(ABC):
    def __init__(self, config: BaseDatasetSamplerConfig):
        self.config = config
        self._sample_calls = 0

    @abstractmethod
    def _score_row(self, row: dict) -> float: ...

    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pure selection core shared by create_sample (the training path) and count_selected (the
        # stats path), so the recorded count is always exactly what training sees. Operates on a
        # copy: callers (e.g. the reconcile DataFrame) must not gain a leaked "score" column.
        df = df.copy()
        df["score"] = df.apply(self._score_row, axis=1)

        if self.config.drop_non_positive:
            df = df[df["score"] > 0]

        df = df.sort_values("score", ascending=False)
        sampled_df = df.head(self.config.top_k)
        # Sort in the increasing order of score so that easier samples are seen first during training (potentially leading to faster convergence, as the model learns from easier examples before harder ones).
        sampled_df = sampled_df.sort_values("score", ascending=True)

        return sampled_df

    def create_sample(self, ds: Dataset) -> Dataset:
        df = cast(pd.DataFrame, ds.to_pandas())
        sampled_df = self._select(df)

        if len(sampled_df) == 0:
            raise RuntimeError(
                f"{type(self).__name__}: all {len(df)} candidate rows scored <= 0 "
                f"(drop_non_positive={self.config.drop_non_positive}); no training samples remain."
            )

        if self.config.shuffle:
            sampled_df = sampled_df.sample(
                frac=1,
                random_state=self.config.shuffle_seed + self._sample_calls,
            )
        self._sample_calls += 1

        sampled_ds = Dataset.from_pandas(sampled_df)
        return sampled_ds

    def count_selected(self, df: pd.DataFrame) -> int:
        # How many rows create_sample would select from df. Never raises (may return 0) so it is
        # safe to call for stats/monitoring even on an epoch that would abort training.
        return len(self._select(df))
