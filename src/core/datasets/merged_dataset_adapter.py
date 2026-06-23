from typing import override

import pandas as pd
from datasets import Dataset, concatenate_datasets

from core.datasets.abstract_dataset_adapter import AbstractDatasetAdapter
from core.datasets.base_dataset_adapter import BaseDatasetAdapter


class MergedDatasetAdapter(AbstractDatasetAdapter):
    def __init__(self, dataset_adapters: list[BaseDatasetAdapter]):
        self.dataset_adapters = dataset_adapters

    @override
    def process_dataset(self, path_override: str | None = None) -> Dataset:
        # Every sub-adapter reads the same source: path_override (e.g. the resampling per-epoch
        # parquet) when given, otherwise each adapter's own configured path. Each applies its own
        # sampler / target builder, then the results are concatenated.
        datasets = [
            adapter.process_dataset(path_override=path_override, strict=True)
            for adapter in self.dataset_adapters
        ]

        ds = concatenate_datasets(datasets)

        return ds

    def sampled_size(self) -> int | None:
        total = 0
        for adapter in self.dataset_adapters:
            size = adapter.sampled_size()
            if size is None:
                return None
            total += size
        return total

    @override
    def selected_sample_counts(self, df: pd.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        for adapter in self.dataset_adapters:
            counts.update(adapter.selected_sample_counts(df))
        return counts

    @override
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None:
        raise NotImplementedError(
            "Saving is not implemented for MergedDatasetAdapter. Please save individual datasets separately."
        )
