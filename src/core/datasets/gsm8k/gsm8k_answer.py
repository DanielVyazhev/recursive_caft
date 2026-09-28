"""Numeric answer parsing for GSM8K.

Gold answers are the text after `####` in the original solution. They are integers, but 93 of them
(79 train / 14 test) carry thousands separators ("1,080", "2,520,000"), and a model may answer with
"$1,080", "1080.", "1080.00" or "**1080**". Comparing raw strings marks all of those wrong, so every
comparison goes through `parse_number` and compares `Decimal` values.

Two parsing modes:

* `parse_number` (strict): the whole text must be one number, optionally wrapped in the decorations
  listed above. Used where the model is expected to emit *only* the number: the answer after
  `</think>` in online entropy estimation, where the entropy is taken over the answer tokens and a
  non-numeric continuation would put the score on the wrong tokens.
* `extract_last_number` (lenient): the last number anywhere in the text. Used as a fallback when
  grading the CoT eval, so "The answer is 72." after `</think>` still counts.
"""

import re
from decimal import Decimal, InvalidOperation

# A number with optional sign, either plain digits or digits grouped by thousands commas
# ("1,080", "2,520,000"), with an optional decimal part. A comma that does not separate a group of
# exactly three digits ("1,5") is not a thousands separator, so it is not matched as one number.
_NUMBER = r"[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-−]?\.\d+"
_NUMBER_RE = re.compile(rf"(?<![\d.,])(?:{_NUMBER})(?![\d])")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_STRICT_RE = re.compile(rf"^(?P<num>{_NUMBER})$")


def _clean(text: str) -> str:
    text = text.strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    # Markdown emphasis and a GSM8K-style "####" marker are decoration, not part of the number.
    text = text.replace("**", "").replace("__", "").replace("####", "")
    text = text.replace("\\$", "$").replace("\\,", ",")
    return text.strip()


def _to_decimal(number: str) -> Decimal | None:
    number = number.replace(",", "").replace("−", "-")
    try:
        value = Decimal(number)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def parse_number(text: str) -> Decimal | None:
    """Strict parse: `text` must be exactly one number, else None.

    Accepted decorations: surrounding whitespace, a currency sign ("$72", "-$5"), thousands commas,
    a single trailing period ("72."), a trailing percent sign, markdown bold, `\\boxed{}`.
    """
    text = _clean(text)
    if text.endswith("."):
        text = text[:-1].rstrip()
    if text.endswith("%"):
        text = text[:-1].rstrip()
    # "$72", "-$5" and "$-5" are all the same number.
    if text.startswith(("-$", "−$")):
        text = "-" + text[2:].lstrip()
    elif text.startswith("$"):
        text = text[1:].lstrip()

    match = _STRICT_RE.match(text)
    if match is None:
        return None
    return _to_decimal(match.group("num"))


def extract_last_number(text: str) -> Decimal | None:
    """Lenient parse: the last number that appears anywhere in `text`, else None."""
    text = _clean(text)
    # "$-5" -> "-5" so the sign stays attached to its number after the currency sign.
    text = text.replace("$-", "-").replace("$−", "-").replace("$", "")
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return _to_decimal(matches[-1])


def format_number(value: Decimal) -> str:
    """Canonical text for a number: no thousands commas, no trailing zeros ("1,080" -> "1080")."""
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def gold_answer(row: dict) -> Decimal:
    """The row's gold answer as a number. Raises ValueError if it is not one (a data error)."""
    value = parse_number(str(row["answer"]))
    if value is None:
        raise ValueError(f"GSM8K gold answer is not a number: {row['answer']!r} (question_id={row.get('question_id')})")
    return value


# Calculator annotations in the original solutions: "48/2 = <<48/2=24>>24".
_CALCULATOR_RE = re.compile(r"<<[^<>]*>>")


def reasoning_steps(row: dict) -> str:
    """The human-written solution without calculator annotations and without the final `#### N`
    line, i.e. the reasoning a student is trained to produce before `</think>`."""
    solution = str(row["answer_w_steps"])
    marker = solution.rfind("####")
    if marker == -1:
        raise ValueError(f"GSM8K solution has no '####' answer line (question_id={row.get('question_id')})")
    return _CALCULATOR_RE.sub("", solution[:marker]).strip()


def solution_final_answer(row: dict) -> Decimal | None:
    """The number after `####` in the original solution."""
    solution = str(row["answer_w_steps"])
    marker = solution.rfind("####")
    if marker == -1:
        return None
    return parse_number(solution[marker + len("####") :])
