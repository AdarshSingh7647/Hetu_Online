"""
Loaders + graders for the four eval benchmarks (AIME24, AIME25, MATH500,
GPQA-Diamond). Each loader returns a list of dicts with a uniform shape:
    {"id": str, "question": str, "reference": str, "choices": Optional[dict]}
"choices" is only set for GPQA (the lettered multiple-choice options actually
shown to the model); every other benchmark leaves it None.

Grading is separated from prompting: `grade_math` does boxed-answer exact
match (with light normalization) for AIME24/25/MATH500, `grade_gpqa` extracts
a letter choice for GPQA-Diamond. Both take the full raw model response text
(not just the final line) and are tolerant of the response containing a
<think>...</think> block ahead of the actual answer.

All four benchmarks are pulled fresh from the Hugging Face Hub via
`datasets.load_dataset` -- nothing here is a local file, so this module has
no state to migrate between machines beyond the (small) HF_HOME cache.
"""

import random
import re
from typing import Optional

BENCHMARKS = ("aime24", "aime25", "math500", "gpqa_diamond")


def load_benchmark(name: str) -> list[dict]:
    if name == "aime24":
        return _load_aime24()
    if name == "aime25":
        return _load_aime25()
    if name == "math500":
        return _load_math500()
    if name == "gpqa_diamond":
        return _load_gpqa_diamond()
    raise ValueError(f"unknown benchmark: {name}")


# `datasets` is imported lazily, inside each loader below (not at module
# level): it's an eval-env-only dependency, and this module's
# BENCHMARKS/format_question/grade are also imported by cli.py just to
# build --benchmark's argparse choices, which must work even in a
# trainer-only environment that never installed `datasets`.


def _load_aime24() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    return [
        {"id": row["ID"], "question": row["Problem"], "reference": str(row["Answer"]), "choices": None}
        for row in ds
    ]


def _load_aime25() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("yentinglin/aime_2025", split="train")
    return [
        {"id": str(row["id"]), "question": row["problem"], "reference": str(row["answer"]), "choices": None}
        for row in ds
    ]


def _load_math500() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {"id": row["unique_id"], "question": row["problem"], "reference": row["answer"], "choices": None}
        for row in ds
    ]


def _load_gpqa_diamond() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rows = []
    for i, row in enumerate(ds):
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Fixed per-question seed so the correct-answer position is
        # deterministic across repeats/checkpoints (not re-shuffled each
        # run, which would make repeats non-comparable) but still
        # decorrelated from dataset order (not always option A).
        rng = random.Random(f"gpqa-{i}")
        order = list(range(4))
        rng.shuffle(order)
        letters = ["A", "B", "C", "D"]
        shuffled = [options[j] for j in order]
        correct_letter = letters[order.index(0)]
        rows.append(
            {
                "id": row["Record ID"],
                "question": row["Question"],
                "reference": correct_letter,
                "choices": dict(zip(letters, shuffled)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Prompt formatting for the question body (model-specific wrapping/system
# prompt happens in model_families.py -- this only builds the
# benchmark-specific question text, identical regardless of which
# checkpoint/mode is used).
# ---------------------------------------------------------------------------

_MATH_DIRECTIVE = "Please reason step by step, and put your final answer within \\boxed{}."


def format_question(benchmark: str, row: dict) -> str:
    if benchmark in ("aime24", "aime25", "math500"):
        return f"{row['question'].strip()}\n\n{_MATH_DIRECTIVE}"
    if benchmark == "gpqa_diamond":
        lines = [row["question"].strip(), ""]
        for letter in ("A", "B", "C", "D"):
            lines.append(f"{letter}) {row['choices'][letter]}")
        lines.append("")
        lines.append(
            "Think step by step, then give your final answer as a single "
            "letter (A, B, C, or D) within \\boxed{}, e.g. \\boxed{A}."
        )
        return "\n".join(lines)
    raise ValueError(f"unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_last_boxed(text: str) -> Optional[str]:
    """Finds the last \\boxed{...} in text and returns its (brace-balanced)
    contents, or None if no \\boxed{ is present."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def _normalize_math_answer(s: str) -> str:
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\text{}", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "")
    s = s.rstrip(".")
    s = s.strip("$")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def _try_numeric_equal(a: str, b: str) -> bool:
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) < 1e-6
    except (ValueError, TypeError):
        return False


def grade_math(response_text: str, reference: str) -> tuple[bool, Optional[str]]:
    """Returns (is_correct, extracted_answer). extracted_answer is None if
    no \\boxed{} was found in the response at all (counted as incorrect,
    not an error -- e.g. the model ran out of generation budget)."""
    extracted = _extract_last_boxed(response_text)
    if extracted is None:
        return False, None
    norm_pred = _normalize_math_answer(extracted)
    norm_ref = _normalize_math_answer(str(reference))
    if norm_pred == norm_ref:
        return True, extracted
    if _try_numeric_equal(norm_pred, norm_ref):
        return True, extracted
    return False, extracted


_LETTER_RE = re.compile(r"\b([A-D])\b")


def grade_gpqa(response_text: str, reference: str) -> tuple[bool, Optional[str]]:
    """Looks for \\boxed{<letter>} first (matches the prompt's own
    instruction); falls back to the last standalone A/B/C/D token in the
    response if no boxed letter is found."""
    boxed = _extract_last_boxed(response_text)
    if boxed is not None:
        m = _LETTER_RE.search(boxed.strip().upper())
        if m:
            letter = m.group(1)
            return letter == reference, letter
    matches = _LETTER_RE.findall(response_text.upper())
    if matches:
        letter = matches[-1]
        return letter == reference, letter
    return False, None


def grade(benchmark: str, response_text: str, reference: str) -> tuple[bool, Optional[str]]:
    if benchmark == "gpqa_diamond":
        return grade_gpqa(response_text, reference)
    return grade_math(response_text, reference)
