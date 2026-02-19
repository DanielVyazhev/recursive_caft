from abc import ABC, abstractmethod

from datasets import Dataset
from transformers.generation.utils import GenerateDecoderOnlyOutput

from core.datasets.abstract_dataset_adapter import TokenizedRow


class BaseComplexityEstimator(ABC):
    @abstractmethod
    def prepare_dataset(self, dataset: Dataset): ...

    @abstractmethod
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
    ): ...
