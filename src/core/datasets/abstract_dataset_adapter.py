from abc import ABC, abstractmethod

import pandas as pd
from datasets import Dataset


class AbstractDatasetAdapter(ABC):
    @abstractmethod
    def process_dataset(self, path_override: str | None = None) -> Dataset: ...

    @abstractmethod
    def save_processed_dataset(self, df: pd.DataFrame, path: str, tmp: bool) -> None: ...

    def sampled_size(self) -> int | None:
        """Row count after sampling without materializing, or None when it can't be known
        without loading the source (e.g. no sampler)."""
        return None
