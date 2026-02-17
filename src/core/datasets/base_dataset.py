from abc import ABC, abstractmethod

from pydraconf import PydraConfig
from transformers import PreTrainedTokenizer


class BaseDatasetConfig(PydraConfig):
    path: str


class BaseDataset(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: BaseDatasetConfig):
        self.tokenizer = tokenizer
        self.config = config

    @abstractmethod
    def system_prompt(self, row: dict) -> str: ...

    @abstractmethod
    def user_prompt(self, row: dict) -> str: ...

    @abstractmethod
    def row_id(self, row: dict) -> str: ...
