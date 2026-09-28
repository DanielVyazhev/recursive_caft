"""GSM8K numeric answer parsing, and the gold data it is applied to.

Gold answers carry thousands commas ("1,080"); a raw string comparison marks "1080" wrong. Every
comparison is numeric, the strict parser accepts only a bare (decorated) number, and the lenient
one takes the last number in free text.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from core.datasets.gsm8k.gsm8k_answer import (
    extract_last_number,
    format_number,
    gold_answer,
    parse_number,
    reasoning_steps,
    solution_final_answer,
)
from core.datasets.gsm8k.gsm8k_validation import validate_gsm8k_data

GSM8K_DIR = Path(__file__).parent.parent / "data/source/gsm8k"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("72", "72"),
        (" 72\n", "72"),
        ("1,080", "1080"),
        ("2,520,000", "2520000"),
        ("$1,080", "1080"),
        ("1080.", "1080"),
        ("1080.00", "1080"),
        ("**72**", "72"),
        ("\\boxed{72}", "72"),
        ("-12", "-12"),
        ("-$5", "-5"),
        ("$-5", "-5"),
        ("−3", "-3"),
        ("0.5", "0.5"),
        ("12.50", "12.5"),
        ("25%", "25"),
    ],
)
def test_strict_parse_accepts_decorated_numbers(text, expected):
    value = parse_number(text)
    assert value is not None
    assert format_number(value) == expected


@pytest.mark.parametrize(
    "text",
    ["", "   ", "abc", "72 dollars", "The answer is 72", "48 + 24 = 72", "1,5", "72 72", "Janet sells 16 - 3"],
)
def test_strict_parse_rejects_anything_but_one_number(text):
    assert parse_number(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is 72.", "72"),
        ("48 + 24 = 72", "72"),
        ("She has $1,080 left.", "1080"),
        ("The answer is $-5", "-5"),
        ("5-3", "3"),
        ("#### 18", "18"),
        ("**Answer:** 1,080", "1080"),
        ("x = \\boxed{42}", "42"),
    ],
)
def test_lenient_parse_takes_the_last_number(text, expected):
    value = extract_last_number(text)
    assert value is not None
    assert format_number(value) == expected


def test_lenient_parse_without_a_number():
    assert extract_last_number("no idea") is None


def test_numeric_equality_ignores_spelling():
    assert parse_number("1,080") == parse_number("1080.00") == Decimal(1080)


def test_reasoning_steps_strip_annotations_and_answer_line():
    row = {
        "question_id": "0",
        "answer": "18",
        "answer_w_steps": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
        "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18",
    }
    assert reasoning_steps(row) == "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day."
    assert solution_final_answer(row) == gold_answer(row) == Decimal(18)


def test_gold_answer_that_is_not_a_number_is_a_data_error():
    with pytest.raises(ValueError):
        gold_answer({"question_id": "x", "answer": "eighteen"})


@pytest.mark.parametrize("split, rows", [("train", 7473), ("test", 1319)])
def test_every_gold_answer_parses_and_matches_its_solution(split, rows):
    df = validate_gsm8k_data(GSM8K_DIR / f"gsm8k_{split}.parquet")
    assert len(df) == rows
