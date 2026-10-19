import pytest

from mtkd.evaluate import (extract_choice, extract_code, extract_math_answer, score_code,
                           score_math, score_qa, score_summ)


@pytest.mark.parametrize("output, expected", [
    ("Natalia sold 24 clips.\n#### 72", 72.0),
    ("#### $1,080", 1080.0),
    ("#### 1080.00", 1080.0),
    ("#### 18 apples", 18.0),
    ("so the total is 3 * 6 = 18 dollars. Final answer: \\boxed{18}", 18.0),
    ("\\boxed{\\$18}", 18.0),
    ("\\boxed{18 \\text{ dollars}}", 18.0),
    ("The answer is 18.\n\nNote: this took 2 steps.", 18.0),
    ("The answer is -5", -5.0),
    ("3 + 4 = 7 and then 7 * 2 = 14", 14.0),
    ("no digits here", None),
])
def test_math_extraction_order(output, expected):
    assert extract_math_answer(output) == expected


def test_math_hashes_beat_later_numbers():
    assert score_math("#### 72\nThat used 2 steps and 3 numbers.", "72") == 1.0


def test_math_boxed_beats_trailing_number():
    assert score_math("x = \\boxed{42}. Check: 42 / 2 = 21", "42") == 1.0


def test_math_truncated_output_scores_zero():
    assert score_math("Step 1: 2 + 2 = 4, then", "18") == 0.0


LABELS = ["A", "B", "C", "D"]


@pytest.mark.parametrize("output, expected", [
    ("B", "B"),
    ("B.", "B"),
    ("(C)", "C"),
    ("C) photosynthesis", "C"),
    ("The answer is B.", "B"),
    ("A plant needs light, so the answer is C", "C"),
    ("Answer: c", "C"),
    ("Option A: wrong because... Option B: also wrong. The correct answer is D.", "D"),
    ("\\boxed{C}", "C"),
    ("the answer is a bit tricky", None),
    ("I am not sure.", None),
])
def test_choice_extraction(output, expected):
    assert extract_choice(output, LABELS) == expected


def test_choice_numeric_labels():
    assert extract_choice("The answer is 3", ["1", "2", "3", "4"]) == "3"
    assert score_qa("2", "2", ["1", "2", "3", "4"]) == 1.0


def test_rouge_identical_is_one_and_reports_length():
    scores = score_summ("Amanda baked cookies.", "Amanda baked cookies.")
    assert scores["rougeL"] == pytest.approx(1.0)
    assert scores["words"] == 3.0


@pytest.mark.parametrize("output", [
    "```py\ndef f(x):\n    return x\n```",
    "```python\ndef f(x):\n    return x\n```",
    "```Python3\ndef f(x):\n    return x\n```",
    "Here you go:\n```\ndef f(x):\n    return x\n```\nThis returns x.",
    "```python\n# usage\nprint(1)\n```\n```python\ndef f(x):\n    return x\n```",
    "```python\ndef f(x):\n    return x\n",
])
def test_code_extraction(output):
    assert extract_code(output).strip() == "def f(x):\n    return x"


def test_code_extraction_ignores_other_languages():
    assert extract_code("```bash\npip install x\n```\n```python\ndef f(): pass\n```").startswith("def f")


def test_score_code_end_to_end():
    reference = {"tests": ["assert f(2) == 4"], "imports": [], "setup": ""}
    assert score_code("```python\ndef f(x):\n    return x * 2\n```", reference) == 1.0
    assert score_code("```python\ndef f(x):\n    return x\n```", reference) == 0.0
