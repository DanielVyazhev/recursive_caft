from abc import ABC, abstractmethod

from core.datasets.causal_dataset import CausalDataset, CausalDatasetConfig


class QADatasetConfig(CausalDatasetConfig):
    pass


class QADataset[C: QADatasetConfig](CausalDataset[C], ABC):
    @abstractmethod
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]: ...
