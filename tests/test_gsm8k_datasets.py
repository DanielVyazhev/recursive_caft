"""GSM8K dataset formatting and grading with the experiment's tokenizer setup.

* training targets `<think>steps</think>N` / `<think></think>N`, tokenized exactly as for training,
  decode back to text that grades as correct (the pre-training validation, on real rows);
* the estimation prompt ends with an empty `<think></think>`;
* online estimation grading is strict: anything but a bare number is a failed measurement;
* CoT eval grading reads only the text after `</think>`.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.datasets.gsm8k.gsm8k_direct_response_dataset import GSM8KDirectResponseDataset
from core.datasets.gsm8k.gsm8k_empty_thinking_response_dataset import GSM8KEmptyThinkingResponseDataset
from core.datasets.gsm8k.gsm8k_reasoning_response_dataset import GSM8KReasoningResponseDataset
from core.datasets.gsm8k.gsm8k_validation import audit_gsm8k_eval, validate_gsm8k_formatting
from core.datasets.qa_dataset import InvalidAnswerError, QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter

GSM8K_DIR = Path(__file__).parent.parent / "data/source/gsm8k"
ROW = {
    "question_id": "7",
    "question": "How many?",
    "answer": "1,080",
    "answer_w_steps": "2 * 540 = <<2*540=1080>>1080\n#### 1,080",
}


def _dataset(cls, tokenizer):
    return cls(config=QADatasetConfig(path="sentinel", dataset_id="x"), tokenizer=tokenizer)


@pytest.mark.parametrize("split", ["train", "test"])
def test_training_targets_round_trip_to_correct_answers(thinking_tokenizer, split):
    path = GSM8K_DIR / f"gsm8k_{split}.parquet"
    df = pd.read_parquet(path)
    # Every row with a comma or negative gold answer, plus a slice of ordinary ones.
    tricky = df[df["answer"].str.contains(",|-", regex=True)]
    rows = pd.concat([tricky, df.head(200)]).to_dict("records")
    validate_gsm8k_formatting(thinking_tokenizer, path, rows)


def test_targets_use_the_canonical_answer(thinking_tokenizer):
    tok = thinking_tokenizer
    assert _dataset(GSM8KDirectResponseDataset, tok).assistant_response(ROW) == "1080"
    assert (
        _dataset(GSM8KEmptyThinkingResponseDataset, tok).assistant_response(ROW)
        == f"{tok.thinking_start_token}{tok.thinking_end_token}1080"
    )
    assert (
        _dataset(GSM8KReasoningResponseDataset, tok).assistant_response(ROW)
        == f"{tok.thinking_start_token}2 * 540 = 1080{tok.thinking_end_token}1080"
    )


def test_estimation_prompt_ends_with_empty_thinking(thinking_tokenizer):
    tok = thinking_tokenizer
    adapter = QADatasetAdapter(dataset=_dataset(GSM8KEmptyThinkingResponseDataset, tok), add_empty_thinking=True)
    plain = QADatasetAdapter(dataset=_dataset(GSM8KEmptyThinkingResponseDataset, tok))
    ids = adapter.process_row(ROW).input_ids
    assert ids[-2:] == [tok.thinking_start_token_id, tok.thinking_end_token_id]
    assert ids[:-2] == plain.process_row(ROW).input_ids


def test_empty_thinking_and_thinking_start_are_exclusive(thinking_tokenizer):
    with pytest.raises(AssertionError):
        QADatasetAdapter(
            dataset=_dataset(GSM8KEmptyThinkingResponseDataset, thinking_tokenizer),
            add_thinking_start_token=True,
            add_empty_thinking=True,
        )


@pytest.mark.parametrize(
    "response, parsed, correct",
    [("1080", "1080", True), ("1,080", "1080", True), ("$1080.", "1080", True), ("1081", "1081", False)],
)
def test_estimation_grading_of_numbers(thinking_tokenizer, response, parsed, correct):
    ds = _dataset(GSM8KEmptyThinkingResponseDataset, thinking_tokenizer)
    assert ds.verify_assistant_response(ROW, response) == (parsed, correct)


@pytest.mark.parametrize("response", ["", "  ", "Let's compute: 2 * 540", "1080 dollars", "First, find the number"])
def test_estimation_grading_rejects_non_numbers_as_failed_measurements(thinking_tokenizer, response):
    ds = _dataset(GSM8KEmptyThinkingResponseDataset, thinking_tokenizer)
    with pytest.raises(InvalidAnswerError):
        ds.verify_assistant_response(ROW, response)


def test_direct_grading_is_numeric_and_never_raises(thinking_tokenizer):
    ds = _dataset(GSM8KDirectResponseDataset, thinking_tokenizer)
    assert ds.verify_assistant_response(ROW, "1080") == ("1080", True)
    assert ds.verify_assistant_response(ROW, "48 + 24 =") == ("48 + 24 =", False)


def test_cot_grading_reads_only_the_answer_after_think(thinking_tokenizer):
    tok = thinking_tokenizer
    ds = _dataset(GSM8KReasoningResponseDataset, tok)
    end, eos = tok.thinking_end_token, tok.eos_token
    assert ds.verify_assistant_response(ROW, f"2 * 540 = 1080{end}1080{eos}") == ("1080", True)
    assert ds.verify_assistant_response(ROW, f"2 * 540 = 1080{end}The answer is $1,080.{eos}") == ("1080", True)
    # The final answer after </think> decides, not numbers in the reasoning.
    assert ds.verify_assistant_response(ROW, f"2 * 540 = 1080{end}1081{eos}") == ("1081", False)
    # No </think>: the reasoning was cut off, so there is no answer.
    assert ds.verify_assistant_response(ROW, "2 * 540 = 1080") == ("", False)
    assert ds.verify_assistant_response(ROW, f"2 * 540{end}no idea{eos}") == ("no idea", False)


def test_eval_audit_regrades_saved_responses(thinking_tokenizer, tmp_path):
    tok = thinking_tokenizer
    end = tok.thinking_end_token
    test = pd.DataFrame([{**ROW, "question_id": str(i)} for i in range(5)])
    test_path = tmp_path / "test.parquet"
    test.to_parquet(test_path)

    responses = pd.DataFrame(
        {
            "row_id": ["0", "1", "2", "3", "4"],
            "response": [f"x{end}1080", f"x{end}The answer is 1,080.", "cut off 1080", f"x{end}???", f"x{end}1081"],
            # Row 1 was stored as incorrect by a grader that compared strings.
            "is_correct": [True, False, False, False, False],
        }
    )
    evals = tmp_path / "run/checkpoint-10/evals/gsm8k_test_cap2048"
    evals.mkdir(parents=True)
    responses.to_parquet(evals / "responses.parquet")

    report = audit_gsm8k_eval(tmp_path / "run", "gsm8k_test_cap2048", test_path, tok, tmp_path / "audit.json")
    counts = report["checkpoint-10"]
    assert counts["correct"] == 2
    assert counts["strict_number_after_think"] == 2
    assert counts["lenient_number_after_think"] == 1
    assert counts["no_think_end"] == 1
    assert counts["no_number_after_think"] == 1
    assert counts["stored_is_correct_disagrees"] == 1
    assert counts["missing_rows"] == 0
    assert (tmp_path / "audit.json").exists()
