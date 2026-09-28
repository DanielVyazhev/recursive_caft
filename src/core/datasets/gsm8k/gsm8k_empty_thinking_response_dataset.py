from typing import override

from core.datasets.gsm8k.gsm8k_answer import format_number, gold_answer, parse_number
from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset
from core.datasets.qa_dataset import InvalidAnswerError


class GSM8KEmptyThinkingResponseDataset(GSM8KDirectResponseDataset):
    """The answer (several tokens) right after an empty `<think></think>`.

    Plays the role MMLUSingleTokenResponseDataset plays for MMLU, in the same two places:

    * online entropy estimation: the prompt ends with `<think></think>` (QADatasetAdapter with
      add_empty_thinking=True), so the model's next tokens are the answer and the entropy is taken
      over them;
    * the small training mix: `<think></think>N` targets keep the student answering in exactly that
      format, so the per-epoch estimation stays measurable as the student becomes CoT-native.

    verify_assistant_response is strict: anything but a bare number is a failed measurement
    (InvalidAnswerError -> NaN entropy -> backfilled by the trainer), never an entropy computed over
    words or over the start of a reasoning chain.
    """

    @override
    def assistant_response(self, row: dict) -> str:
        empty_thinking = f"{self.tokenizer.thinking_start_token}{self.tokenizer.thinking_end_token}"
        return f"{empty_thinking}{format_number(gold_answer(row))}"

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        response = assistant_response
        # The estimation runner decodes with skip_special_tokens=True; strip them anyway so the
        # check also holds for text decoded with them kept.
        for tok in self.tokenizer.all_special_tokens:
            response = response.replace(tok, "")
        response = response.strip()

        if not response:
            raise InvalidAnswerError("empty answer after </think>")
        value = parse_number(response)
        if value is None:
            raise InvalidAnswerError(f"answer after </think> is not a single number: {response[:80]!r}")
        return format_number(value), value == gold_answer(row)
