"""Checks that GSM8K answers are formatted and graded correctly.

`validate_gsm8k_pipeline` runs before training (GSM8K experiments call it first thing) and raises on
any problem, so a run never starts with answers that cannot be matched:

* data: unique question ids, every gold answer is a number, the `#### N` line of every solution
  equals the gold answer, the cleaned reasoning is non-empty and carries no annotations;
* round trip, with the experiment's own tokenizer and chat template: every training target
  (`<think>steps</think>N` and `<think></think>N`) is tokenized exactly as for training, its
  supervised tokens are decoded back and graded, and must come out correct;
* estimation prompts end with `<think></think>`, and every accepted answer spelling of the gold
  answer ("1080", "1,080", "$1,080", "1080.", "1080.00", "**1080**") grades as correct.

`audit_gsm8k_eval` runs after the CoT eval and re-grades every saved response per checkpoint,
reporting how answers were parsed (strict / lenient / missing `</think>` / unparseable) and whether
the stored `is_correct` agrees with the re-grade.
"""

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.gsm8k.gsm8k_answer import (
    extract_last_number,
    format_number,
    gold_answer,
    parse_number,
    reasoning_steps,
    solution_final_answer,
)
from core.datasets.gsm8k.gsm8k_empty_thinking_response_dataset import GSM8KEmptyThinkingResponseDataset
from core.datasets.gsm8k.gsm8k_reasoning_response_dataset import GSM8KReasoningResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.utils.logger import logger

_MAX_REPORTED = 5


def _answer_spellings(value: Decimal) -> list[str]:
    plain = format_number(value)
    spellings = [plain, f"{plain}.", f"**{plain}**", f"\\boxed{{{plain}}}", f"${plain}"]
    if value == value.to_integral_value():
        grouped = f"{int(value):,}"
        spellings += [grouped, f"${grouped}", f"{plain}.00"]
    return spellings


def _fail(problems: dict[str, list[str]]) -> None:
    lines = [f"{name}: {len(ids)} rows, e.g. {ids[:_MAX_REPORTED]}" for name, ids in problems.items() if ids]
    if lines:
        raise ValueError("GSM8K answer validation failed:\n  " + "\n  ".join(lines))


def validate_gsm8k_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    problems: dict[str, list[str]] = {
        "duplicate question_id": df.loc[df["question_id"].duplicated(), "question_id"].astype(str).tolist(),
        "gold answer is not a number": [],
        "#### answer != gold answer": [],
        "empty or unclean reasoning": [],
    }
    for row in df.to_dict("records"):
        qid = str(row["question_id"])
        gold = parse_number(str(row["answer"]))
        if gold is None:
            problems["gold answer is not a number"].append(qid)
            continue
        if solution_final_answer(row) != gold:
            problems["#### answer != gold answer"].append(qid)
        try:
            steps = reasoning_steps(row)
        except ValueError:
            steps = ""
        if not steps or "<<" in steps or ">>" in steps or "####" in steps:
            problems["empty or unclean reasoning"].append(qid)
    _fail(problems)
    return df


def validate_gsm8k_formatting(tokenizer: PreTrainedTokenizer, path: str | Path, rows: list[dict]) -> None:
    config = QADatasetConfig(path=str(path), dataset_id="gsm8k_validation")
    reasoning = GSM8KReasoningResponseDataset(tokenizer=tokenizer, config=config)
    empty_thinking = GSM8KEmptyThinkingResponseDataset(tokenizer=tokenizer, config=config)
    estimation_adapter = QADatasetAdapter(dataset=empty_thinking, add_empty_thinking=True)
    empty_thinking_suffix = [tokenizer.thinking_start_token_id, tokenizer.thinking_end_token_id]

    problems: dict[str, list[str]] = {
        "reasoning target does not grade as correct": [],
        "<think></think> target does not grade as correct": [],
        "estimation prompt does not end with <think></think>": [],
        "accepted answer spelling does not grade as correct": [],
    }
    for row in rows:
        qid = str(row["question_id"])
        gold = gold_answer(row)
        expected = format_number(gold)

        for dataset, name in (
            (reasoning, "reasoning target does not grade as correct"),
            (empty_thinking, "<think></think> target does not grade as correct"),
        ):
            tokenized = CausalDatasetAdapter(dataset=dataset).process_row(row)
            supervised = [tok for tok in tokenized.labels if tok != -100]
            decoded = tokenizer.decode(supervised, skip_special_tokens=False)
            # The CoT eval grades the generation, which starts right after the prompt's <think>; the
            # supervised span starts with that <think>, so grading it is the same check.
            parsed, correct = dataset.verify_assistant_response(row, decoded)
            if not correct or parsed != expected:
                problems[name].append(f"{qid}: {decoded[-60:]!r} -> {parsed!r}")

        prompt = estimation_adapter.process_row(row)
        if prompt.input_ids[-2:] != empty_thinking_suffix:
            problems["estimation prompt does not end with <think></think>"].append(qid)

        for spelling in _answer_spellings(gold):
            graded = [
                empty_thinking.verify_assistant_response(row, spelling)[1],
                reasoning.verify_assistant_response(row, f"{tokenizer.thinking_end_token}{spelling}")[1],
                reasoning.verify_assistant_response(row, f"{tokenizer.thinking_end_token}The answer is {spelling}")[1],
            ]
            if not all(graded):
                problems["accepted answer spelling does not grade as correct"].append(f"{qid}: {spelling!r}")
    _fail(problems)


def validate_gsm8k_pipeline(tokenizer: PreTrainedTokenizer, paths: list[str | Path]) -> None:
    for path in paths:
        df = validate_gsm8k_data(path)
        validate_gsm8k_formatting(tokenizer, path, df.to_dict("records"))
        logger.info(f"GSM8K answer validation passed for {len(df)} rows of {path}")


def audit_gsm8k_eval(
    checkpoints_dir: str | Path,
    eval_dataset_id: str,
    test_path: str | Path,
    tokenizer: PreTrainedTokenizer,
    out_file: str | Path,
) -> dict:
    dataset = GSM8KReasoningResponseDataset(
        tokenizer=tokenizer, config=QADatasetConfig(path=str(test_path), dataset_id=eval_dataset_id)
    )
    test = {str(row["question_id"]): row for row in pd.read_parquet(test_path).to_dict("records")}

    report: dict[str, dict] = {}
    checkpoint_dirs = sorted(Path(checkpoints_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    for ckpt_dir in checkpoint_dirs:
        responses_path = ckpt_dir / "evals" / eval_dataset_id / "responses.parquet"
        if not responses_path.exists():
            continue
        responses = pd.read_parquet(responses_path)
        counts = {
            "total": len(responses),
            "correct": 0,
            "strict_number_after_think": 0,
            "lenient_number_after_think": 0,
            "no_think_end": 0,
            "no_number_after_think": 0,
            "stored_is_correct_disagrees": 0,
            "missing_rows": len(test) - responses["row_id"].astype(str).nunique(),
        }
        for record in responses.to_dict("records"):
            row = test[str(record["row_id"])]
            answer_text = dataset.answer_text(record["response"])
            if answer_text is None:
                counts["no_think_end"] += 1
            elif parse_number(answer_text) is not None:
                counts["strict_number_after_think"] += 1
            elif extract_last_number(answer_text) is not None:
                counts["lenient_number_after_think"] += 1
            else:
                counts["no_number_after_think"] += 1

            _, correct = dataset.verify_assistant_response(row, record["response"])
            counts["correct"] += int(correct)
            counts["stored_is_correct_disagrees"] += int(bool(record["is_correct"]) != correct)
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else 0.0
        report[ckpt_dir.name] = counts

        if counts["stored_is_correct_disagrees"] or counts["missing_rows"]:
            logger.warning(f"GSM8K eval audit {ckpt_dir.name}: {counts}")
        else:
            logger.info(f"GSM8K eval audit {ckpt_dir.name}: {counts}")

    Path(out_file).write_text(json.dumps(report, indent=2))
    return report
