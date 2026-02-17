from abc import ABC, abstractmethod

from core.datasets.causal_dataset import CausalDataset


class QADataset(CausalDataset, ABC):
    @abstractmethod
    def verify_assistant_response(self, row: dict, assistant_response: str) -> bool: ...
