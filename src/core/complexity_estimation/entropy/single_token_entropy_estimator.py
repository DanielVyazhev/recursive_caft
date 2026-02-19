from typing import override

from transformers.generation.utils import GenerateDecoderOnlyOutput

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.datasets.abstract_dataset_adapter import TokenizedRow


class SingleTokenEntropyEstimator(BaseComplexityEstimator):
    def __init__(self, entropy_column_name: str = "entropy_value") -> None:
        super().__init__()
        self.entropy_column_name = entropy_column_name

    @override
    def prepare_dataset(self, dataset):
        if self.entropy_column_name not in dataset.column_names:
            dataset = dataset.add_column(self.entropy_column_name, [None] * len(dataset))

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
    ):
        input_length = len(input.input_ids)
        first_token_logits = outputs.scores[input_length][0]
        entropy = compute_entropy_from_logits(first_token_logits)
        dataset_row[self.entropy_column_name] = entropy
