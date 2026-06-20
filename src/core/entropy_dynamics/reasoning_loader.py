"""
Load teacher reasoning chains from parquet files.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizer


@dataclass
class TeacherReasoning:
    question_id: str
    question: str
    options: list[str]          # ["text_a", "text_b", ...]
    gold_answer: str            # letter, e.g. "a"
    thinking_text: str          # raw reasoning string
    thinking_token_ids: list[int]  # tokenized reasoning


def load_teacher_reasoning(
    path: str | Path,
    tokenizer: PreTrainedTokenizer,
    min_thinking_tokens: int = 16,
    max_thinking_tokens: int | None = None,
) -> list[TeacherReasoning]:
    """Load and tokenize teacher reasoning chains.

    min_thinking_tokens: drop chains shorter than this (too short to chunk).
    max_thinking_tokens: if set, drop chains LONGER than this. Use it to keep
        the full prompt within the model's max_model_len and to avoid wasting
        inference on degenerate (looping) ultra-long chains. Set it to roughly
        (max_model_len - ~512) so the prompt overhead still fits.
    """
    df = pd.read_parquet(path)

    # Маппим distill_reasoning в ожидаемый колонку thinking, если пришел новый датасет
    if "distill_reasoning" in df.columns and "thinking" not in df.columns:
        df = df.rename(columns={"distill_reasoning": "thinking"})

    if "input" in df.columns and "output" in df.columns:
        records = _parse_synth_aug(df)
    elif "thinking" in df.columns:  # Сюда теперь зайдет и ваш датасет
        records = _parse_flat(df)
    else:
        raise ValueError(
            f"Unrecognised parquet schema. "
            f"Expected either (input, output) or (question_id, question, options, answer, thinking). "
            f"Got columns: {list(df.columns)}"
        )

    # --- tokenize and filter ---
    results: list[TeacherReasoning] = []
    n_too_short = 0
    n_too_long = 0
    for rec in records:
        if not rec.thinking_text or not rec.thinking_text.strip():
            continue

        token_ids = tokenizer.encode(rec.thinking_text, add_special_tokens=False)
        if len(token_ids) < min_thinking_tokens:
            n_too_short += 1
            continue
        if max_thinking_tokens is not None and len(token_ids) > max_thinking_tokens:
            n_too_long += 1
            continue

        rec.thinking_token_ids = token_ids
        results.append(rec)

    if n_too_long:
        print(
            f"Dropped {n_too_long} chains exceeding max_thinking_tokens={max_thinking_tokens} "
            f"(kept {len(results)} / {len(results) + n_too_long + n_too_short})."
        )

    return results


def _parse_synth_aug(df: pd.DataFrame) -> list[TeacherReasoning]:
    records: list[TeacherReasoning] = []

    for _, row in df.iterrows():
        inp = row["input"]
        out = row["output"]

        if not isinstance(inp, dict) or not isinstance(out, dict):
            continue
        if "error" in out and out["error"] is not None:
            continue

        thinking = out.get("thinking", "") or ""
        if not thinking:
            continue

        raw_options = inp.get("options", [])
        if isinstance(raw_options, dict):
            options_list = list(raw_options.values())
        elif isinstance(raw_options, list):
            options_list = raw_options
        else:
            options_list = _safe_literal_eval(raw_options)

        records.append(TeacherReasoning(
            question_id=str(inp.get("question_id", "")),
            question=str(inp.get("question", "")),
            options=options_list,
            gold_answer=str(inp.get("gold", out.get("answer", ""))).strip().lower(),
            thinking_text=thinking,
            thinking_token_ids=[],  # filled later
        ))

    return records


def _parse_flat(df: pd.DataFrame) -> list[TeacherReasoning]:
    records: list[TeacherReasoning] = []

    for _, row in df.iterrows():
        thinking = str(row.get("thinking", "") or "")
        if not thinking:
            continue

        raw_options = row.get("options", "[]")
        if isinstance(raw_options, list):
            options_list = raw_options
        else:
            options_list = _safe_literal_eval(raw_options)

        records.append(TeacherReasoning(
            question_id=str(row.get("question_id", "")),
            question=str(row.get("question", "")),
            options=options_list,
            gold_answer=str(row.get("answer", "")).strip().lower(),
            thinking_text=thinking,
            thinking_token_ids=[],
        ))

    return records


def _safe_literal_eval(s) -> list:
    try:
        result = ast.literal_eval(str(s))
        return list(result) if isinstance(result, (list, tuple)) else []
    except Exception:
        return []
