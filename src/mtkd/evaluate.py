"""Per-category scorers.

math: GSM8K exact match. Extraction order: '####', then the last \\boxed{}, then the last
      "answer is ...", then the last number in the output.
qa:   ARC-Challenge. The headline metric is letter log-likelihood (gen.choice_predictions);
      extract_choice scores free-form generations as a secondary metric.
summ: SamSum ROUGE-1/2/L F1 (ROUGE-L is the headline) plus output length in words.
code: MBPP pass@1 (greedy), run in a sandbox with a sentinel (sandbox.py).
"""

import re

from rouge_score import rouge_scorer

from mtkd.sandbox import run_tests

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_NUM_RE = re.compile(r"-?\$?\s*\d[\d,]*(?:\.\d+)?")


def _to_float(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _first_number(text: str) -> str | None:
    match = _NUM_RE.search(text)
    return match.group(0) if match else None


def _last_boxed(text: str) -> str | None:
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    depth = 1
    for i in range(starts[-1], len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[starts[-1]:i]
    return text[starts[-1]:]


def extract_math_answer(output: str) -> float | None:
    hashes = re.search(r"####\s*(.*)", output)
    if hashes and (num := _first_number(hashes.group(1))) is not None:
        return _to_float(num)
    boxed = _last_boxed(output)
    if boxed is not None and (num := _first_number(boxed)) is not None:
        return _to_float(num)
    phrases = re.finditer(r"answer\s*(?:is|:|=)\s*:?\s*(.{0,40})", output, re.IGNORECASE)
    for phrase in reversed(list(phrases)):
        if (num := _first_number(phrase.group(1))) is not None:
            return _to_float(num)
    nums = _NUM_RE.findall(output)
    return _to_float(nums[-1]) if nums else None


def score_math(output: str, reference: str) -> float:
    predicted = extract_math_answer(output)
    expected = _to_float(reference)
    if predicted is None or expected is None:
        return 0.0
    return float(abs(predicted - expected) < 1e-6)


_ANSWER_PREFIX = r"(?i:answer)\s*(?i:is)?\s*[:\-]?\s*(?i:option\s*)?\(?"


def extract_choice(output: str, labels: list[str]) -> str | None:
    """Free-form multiple-choice extraction. An explicit "answer is X" anywhere wins over a
    leading letter, so "A plant ... the answer is C" gives C. Lowercase letters count only
    when nothing follows them ("Answer: c"), so "the answer is a bit tricky" gives None."""
    text = output.strip()
    upper = "|".join(re.escape(label) for label in labels)
    lower = "|".join(re.escape(label.lower()) for label in labels if label.isalpha())
    patterns = [re.compile(rf"{_ANSWER_PREFIX}({upper})\)?(?![A-Za-z0-9])")]
    if lower:
        patterns.append(re.compile(rf"{_ANSWER_PREFIX}({lower})\)?(?=\s*$|[.)\]:,;])"))
    hits = [(m.start(), m.group(1)) for pattern in patterns for m in pattern.finditer(text)]
    if hits:
        return _canonical(max(hits)[1], labels)
    boxed = _last_boxed(text)
    if boxed is not None and (label := _canonical(boxed.strip(), labels)):
        return label
    leading = re.match(rf"^\(?({upper})\)?(?:[.):]|\s*$)", text)
    if leading:
        return leading.group(1)
    return _canonical(text.rstrip("."), labels) if len(text) <= 2 else None


def _canonical(token: str, labels: list[str]) -> str | None:
    for label in labels:
        if token.upper() == label.upper():
            return label
    return None


def score_qa(output: str, answer: str, labels: list[str]) -> float:
    return float(extract_choice(output, labels) == answer)


def score_summ(output: str, reference: str) -> dict[str, float]:
    scores = _ROUGE.score(reference, output)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "words": float(len(output.split())),
    }


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|$)", re.DOTALL)
_PY_TAGS = {"", "python", "py", "python3"}


def extract_code(output: str) -> str:
    """First fenced Python block that defines a function or class; else the first Python
    block; else the raw output."""
    blocks = [body for tag, body in _FENCE.findall(output) if tag.lower() in _PY_TAGS]
    for body in blocks:
        if re.search(r"^\s*(def|class)\s", body, re.MULTILINE):
            return body
    return blocks[0] if blocks else output


def score_code(output: str, reference: dict, timeout_s: float = 10.0) -> float:
    result = run_tests(
        extract_code(output),
        tests=reference["tests"],
        imports=reference.get("imports", []),
        setup=reference.get("setup", ""),
        timeout_s=timeout_s,
    )
    return float(result.passed)


def score_generation(category: str, output: str, reference) -> dict[str, float]:
    """Metrics for one generated output. "score" is the headline metric, except for qa,
    whose headline comes from log-likelihood and is added by the caller."""
    if category == "math":
        return {"score": score_math(output, reference)}
    if category == "qa":
        return {"gen_score": score_qa(output, reference["answer"], reference["labels"])}
    if category == "summ":
        rouge = score_summ(output, reference)
        return {"score": rouge["rougeL"], **rouge}
    if category == "code":
        return {"score": score_code(output, reference)}
    raise ValueError(f"unknown category: {category}")
