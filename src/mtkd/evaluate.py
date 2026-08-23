"""Per-category scoring: math exact-match, qa choice-match, summ ROUGE-L, code test-exec."""

import re
import subprocess
import sys
import tempfile

from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score_math(output: str, reference: str) -> float:
    match = re.search(r"####\s*([-+]?[\d,]*\.?\d+)", output)
    if match is None:
        nums = re.findall(r"[-+]?[\d,]*\.?\d+", output)
        if not nums:
            return 0.0
        candidate = nums[-1]
    else:
        candidate = match.group(1)
    try:
        return float(float(candidate.replace(",", "")) == float(reference.replace(",", "")))
    except ValueError:
        return 0.0


def score_qa(output: str, reference: str) -> float:
    match = re.search(r"\b([A-E1-5])\b", output.strip())
    return float(match is not None and match.group(1) == reference)


def score_summ(output: str, reference: str) -> float:
    return _ROUGE.score(reference, output)["rougeL"].fmeasure


def score_code(output: str, reference: str, timeout_s: int = 10) -> float:
    """Extract a code block and run it against the test assertions. Untrusted-code
    execution — subprocess with timeout, no network guarantees. Skip with run_code=False."""
    match = re.search(r"```(?:python)?\n(.*?)```", output, re.DOTALL)
    code = match.group(1) if match else output
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + reference + "\n")
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout_s
        )
        return float(proc.returncode == 0)
    except subprocess.TimeoutExpired:
        return 0.0


def score(category: str, output: str, reference: str, run_code: bool = True) -> float:
    if category == "math":
        return score_math(output, reference)
    if category == "qa":
        return score_qa(output, reference)
    if category == "summ":
        return score_summ(output, reference)
    if category == "code":
        return score_code(output, reference) if run_code else 0.0
    raise ValueError(f"unknown category: {category}")


def aggregate(rows: list[dict]) -> dict:
    """rows: [{"category": str, "score": float}] -> per-category + overall means."""
    by_cat: dict[str, list[float]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row["score"])
    result = {cat: sum(v) / len(v) for cat, v in sorted(by_cat.items())}
    all_scores = [row["score"] for row in rows]
    result["overall"] = sum(all_scores) / len(all_scores)
    return result
