"""AnswerTokensEntropyWithRandomEstimator through the real ComplexityEstimationRunner, with the GSM8K
estimation setup (prompt ending in `<think></think>`, strict numeric grading):

* `entropy_value` aggregates the per-step entropies of the answer tokens only (the end-of-turn
  token and a trailing newline are not scored), next to a fresh `random_value`;
* a row whose continuation is not a bare number is a failed measurement (NaN entropy), which the
  trainer backfills and the samplers drop -- it is never scored on the words it produced;
* the resulting parquet is directly scorable by the entropy-gain sampler.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from core.complexity_estimation.complexity_estimation_runner import (
    ComplexityEstimationRunner,
    ComplexityEstimationRunnerConfig,
    ModelGenerateConfig,
)
from core.complexity_estimation.entropy.answer_tokens_entropy_with_random_estimator import (
    AnswerTokensEntropyWithRandomEstimator,
)
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.datasets.gsm8k.gsm8k_empty_thinking_response_dataset import GSM8KEmptyThinkingResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter


def _peaked_logits(vocab: int, token_id: int, peak: float) -> torch.Tensor:
    logits = torch.zeros(1, vocab)
    logits[0, token_id] = peak
    return logits


class _FakeModel:
    """Emits the tokens of a per-question reply, a newline and the end-of-turn token. Each answer
    token gets a different peak, so its entropy differs; the newline and end-of-turn steps are flat
    (maximal entropy) and must not be scored."""

    def __init__(self, tokenizer, replies: dict[str, str]):
        self.device = "cpu"
        self._tokenizer = tokenizer
        self._replies = replies
        self._vocab = len(tokenizer)
        self.expected: dict[str, list[float]] = {}

    def generate(self, input_ids, attention_mask, **kwargs):
        prompt = self._tokenizer.decode(input_ids[0])
        qid = next(q for q in self._replies if f"question {q}?" in prompt)
        answer_ids = self._tokenizer.encode(self._replies[qid], add_special_tokens=False)
        newline_ids = self._tokenizer.encode("\n", add_special_tokens=False)
        eos = self._tokenizer.convert_tokens_to_ids(self._tokenizer.eos_token)

        scores = [_peaked_logits(self._vocab, tid, 5.0 + i) for i, tid in enumerate(answer_ids)]
        self.expected[qid] = [float(compute_entropy_from_logits(s[0])) for s in scores]
        # A trailing newline and the end-of-turn token, both flat (maximal entropy), not scored.
        scores += [torch.zeros(1, self._vocab)] * (len(newline_ids) + 1)

        appended = torch.tensor([answer_ids + newline_ids + [eos]], dtype=input_ids.dtype)
        return SimpleNamespace(sequences=torch.cat([input_ids, appended], dim=1), scores=tuple(scores))


def _run(tokenizer, tmp_path, replies, aggregation="max"):
    rows = [
        {
            "question_id": qid,
            "question": f"question {qid}?",
            "answer": "1,080",
            "answer_w_steps": "2 * 540 = <<2*540=1080>>1080\n#### 1,080",
            "teacher_entropy": 0.0,
        }
        for qid in replies
    ]
    source = tmp_path / "gsm8k_teacher_entropy.parquet"
    pd.DataFrame(rows).to_parquet(source)

    adapter = QADatasetAdapter(
        dataset=GSM8KEmptyThinkingResponseDataset(
            config=QADatasetConfig(path=str(source), dataset_id="gsm8k_teacher_entropy"), tokenizer=tokenizer
        ),
        add_empty_thinking=True,
    )
    out_path = tmp_path / "out" / "gsm8k_teacher_entropy.parquet"
    model = _FakeModel(tokenizer, replies)
    ComplexityEstimationRunner(
        config=ComplexityEstimationRunnerConfig(
            out_path=str(out_path),
            answer_field_name="estimation_phase_answer",
            answer_correctness_field_name="estimation_phase_answer_correctness",
            generate_config=ModelGenerateConfig(max_new_tokens=16),
            save_every=1000,
        ),
        complexity_estimator=AnswerTokensEntropyWithRandomEstimator(aggregation=aggregation),
    ).estimate(dataset_adapter=adapter, model=model)
    return pd.read_parquet(out_path).set_index("question_id"), model.expected


@pytest.mark.parametrize(
    "aggregation, reduce", [("max", max), ("mean", lambda v: sum(v) / len(v)), ("first", lambda v: v[0])]
)
def test_entropy_is_aggregated_over_answer_tokens_only(thinking_tokenizer, tmp_path, aggregation, reduce):
    res, expected = _run(thinking_tokenizer, tmp_path, {"a": "1080", "b": "1081"}, aggregation)

    for qid in ("a", "b"):
        assert res.loc[qid, "answer_token_count"] == len(expected[qid])
        assert res.loc[qid, "entropy_value"] == pytest.approx(reduce(expected[qid]))
        assert list(res.loc[qid, "answer_token_entropies"]) == pytest.approx(expected[qid])
        assert 0.0 <= res.loc[qid, "random_value"] < 1.0
    assert res.loc["a", "estimation_phase_answer_correctness"]
    assert not res.loc["b", "estimation_phase_answer_correctness"]


def test_non_numeric_continuation_is_a_failed_measurement(thinking_tokenizer, tmp_path):
    res, _ = _run(thinking_tokenizer, tmp_path, {"a": "1080", "b": "First, find the", "c": "1,080"})

    assert pd.isna(res.loc["b", "entropy_value"])
    assert res["entropy_value"].drop("b").notna().all()

    # Scorable by the entropy-gain sampler: the unmeasured row is dropped, the others stay eligible
    # (teacher entropy 0, so every measured row has a positive gain).
    res = res.reset_index()
    assert EntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=10)).count_selected(res) == 2
