from abc import ABC, abstractmethod

from core.datasets.base_dataset import BaseDataset


class CausalDataset(BaseDataset, ABC):
    @abstractmethod
    def assistant_response(self, row: dict) -> str: ...
