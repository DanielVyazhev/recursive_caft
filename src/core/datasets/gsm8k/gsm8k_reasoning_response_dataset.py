from typing import override

from core.datasets.gsm8k.gsm8k_answer import (
    extract_last_number,
    format_number,
    gold_answer,
    parse_number,
    reasoning_steps,
)
from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset


class GSM8KReasoningResponseDataset(GSM8KDirectResponseDataset):
    """`<think>{solution steps}</think>{answer}`, the GSM8K counterpart of MMLUReasoningResponseDataset.

    Training target: the human-written GSM8K solution (`answer_w_steps`) without its `<<a*b=c>>`
    calculator annotations and without the final `#### N` line, then the canonical answer.

    Grading (CoT eval): only the text after the first `</think>` counts. It is parsed strictly first
    (a bare number, the trained format) and, failing that, leniently (the last number in it, e.g.
    "The answer is 72."). No `</think>` at all -> no answer -> incorrect: numbers inside the reasoning
    are intermediate results, not answers.
    """

    @override
    def assistant_response(self, row: dict) -> str:
        return (
            f"{self.tokenizer.thinking_start_token}{reasoning_steps(row)}"
            f"{self.tokenizer.thinking_end_token}{format_number(gold_answer(row))}"
        )

    @override
    def verify_assistant_response(self, row: dict, assistant_response: str) -> tuple[str, bool]:
        answer_text = self.answer_text(assistant_response)
        if answer_text is None:
            return "", False

        value = parse_number(answer_text)
        if value is None:
            value = extract_last_number(answer_text)
        if value is None:
            return answer_text[:80], False
        return format_number(value), value == gold_answer(row)

    def answer_text(self, assistant_response: str) -> str | None:
        """The text after the first `</think>` with special tokens removed, or None without `</think>`."""
        end_token = self.tokenizer.thinking_end_token
        assert isinstance(end_token, str), (
            "Tokenizer must have a thinking_end_token; call setup_thinking_tokens(tokenizer) first."
        )
        position = assistant_response.find(end_token)
        if position == -1:
            return None
        raw = assistant_response[position + len(end_token) :]
        for tok in self.tokenizer.all_special_tokens:
            raw = raw.replace(tok, "")
        return raw.strip()
