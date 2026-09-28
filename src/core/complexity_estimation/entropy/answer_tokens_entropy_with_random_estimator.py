from random import random
from typing import Literal, override

import torch
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import PreTrainedTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.datasets.base_dataset_adapter import TokenizedRow
from core.datasets.qa_dataset import InvalidAnswerError


class AnswerTokensEntropyWithRandomSchema(BaseModel):
    entropy_value: float
    random_value: float
    answer_token_entropies: list[float]
    answer_token_count: int


class AnswerTokensEntropyWithRandomEstimator(BaseComplexityEstimator[AnswerTokensEntropyWithRandomSchema]):
    """Entropy of a multi-token answer, aggregated into the single `entropy_value` the resampling
    trainer and the samplers consume, plus a fresh `random_value` draw per row (see
    SingleTokenEntropyWithRandomEstimator).

    Per-step entropies are taken over the generated answer tokens only: steps that emit a special
    token (EOS/end-of-turn, `<think>`, `</think>`, padding) or a whitespace-only token (a trailing
    newline) are skipped, so the aggregate is never taken on the decision to stop. This matches
    MultiTokenEntropyEstimator, which produced the offline proxy entropies, up to that estimator
    skipping only a trailing EOS (the Qwen2.5-72B proxy answers are bare numbers).

    `aggregation` must match the column the proxy's `teacher_entropy` was taken from (e.g. "max" ->
    the proxy's `max_entropy`), so that the entropy gain compares like with like.
    """

    def __init__(self, aggregation: Literal["max", "mean", "first"] = "max"):
        self.aggregation = aggregation

    @property
    @override
    def schema(self) -> dict[str, FieldInfo]:
        return AnswerTokensEntropyWithRandomSchema.model_fields

    @override
    def estimate_row(
        self,
        dataset_row: dict,
        input: TokenizedRow,
        outputs: GenerateDecoderOnlyOutput,
        parsed_answer: str,
        answer_correctness: bool,
        tokenizer: PreTrainedTokenizer,
    ) -> AnswerTokensEntropyWithRandomSchema:
        assert outputs.scores is not None, "outputs.scores is None — generation must return scores"
        generated_token_ids = outputs.sequences[0].tolist()[len(input.input_ids) :]
        special_ids = set(tokenizer.all_special_ids)

        answer_logits = [
            outputs.scores[step][0]
            for step, token_id in enumerate(generated_token_ids[: len(outputs.scores)])
            if token_id not in special_ids and tokenizer.decode([token_id]).strip()
        ]
        if not answer_logits:
            raise InvalidAnswerError("no answer tokens were generated")

        entropies = compute_entropy_from_logits(torch.stack(answer_logits, dim=0).float())
        if self.aggregation == "max":
            value = entropies.max()
        elif self.aggregation == "mean":
            value = entropies.mean()
        else:
            value = entropies[0]

        return AnswerTokensEntropyWithRandomSchema(
            entropy_value=float(value),
            random_value=random(),
            answer_token_entropies=[float(e) for e in entropies],
            answer_token_count=len(answer_logits),
        )
