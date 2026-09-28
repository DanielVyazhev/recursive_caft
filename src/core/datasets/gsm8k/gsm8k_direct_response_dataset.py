from typing import override

from transformers import PreTrainedTokenizer

from core.datasets.gsm8k.gsm8k_answer import format_number, gold_answer, parse_number
from core.datasets.qa_dataset import QADataset, QADatasetConfig


class GSM8KDirectResponseDataset(QADataset[QADatasetConfig]):
    """The model answers with the number only, no reasoning.

    The prompt is the one the offline proxy entropies (`data/out/single_token_entropy*/gsm8k_*`) were
    measured with; keep it unchanged so online student entropies stay comparable to them.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: QADatasetConfig):
        super().__init__(tokenizer, config)

    @override
    def system_prompt(self, row: dict) -> str:
        return "The following are grade school math word problems. Please, return your answer as a single number (without extra/special symbols) and nothing else."

    @override
    def user_prompt(self, row: dict) -> str:
        question = row["question"]
        return question.strip()

    @override
    def assistant_response(self, row: dict) -> str:
        # Canonical form ("1,080" -> "1080"): the prompt asks for no extra symbols.
        return format_number(gold_answer(row))

    @override
    def row_id(self, row: dict) -> str:
        return str(row["question_id"])

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        # Numeric comparison, so "1080", "1,080" and "$1080." all match a gold "1,080".
        parsed_answer = assistant_response.strip()
        value = parse_number(parsed_answer)
        if value is None:
            return parsed_answer, False
        return format_number(value), value == gold_answer(row)
