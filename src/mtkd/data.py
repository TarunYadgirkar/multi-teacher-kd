"""Benchmark loaders with pinned revisions and disjoint train / probe / eval splits.

| category | train                          | probe (teacher selection)      | eval                  |
|----------|--------------------------------|--------------------------------|-----------------------|
| math     | GSM8K train minus last 300     | GSM8K train, last 300 rows     | GSM8K test            |
| code     | MBPP full train (ids 601-974)  | MBPP full validation + prompt  | MBPP sanitized test   |
| qa       | ARC-Challenge train            | ARC-Challenge validation       | ARC-Challenge test    |
| summ     | SamSum train                   | SamSum validation              | SamSum test           |

Teacher selection never touches a test split. Every example is a plain dict:
{"id", "category", "prompt": [messages], "target": gold completion, "reference": ...}.
"""

import random
import re
from collections.abc import Callable

from datasets import load_dataset

CATEGORIES = ("math", "code", "qa", "summ")

REVISIONS = {
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google-research-datasets/mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "knkarthick/samsum": "6b929ff10edec703164e3ddb2e94aae058c9ab5f",
}

MATH_INSTRUCTION = "Solve the problem. End with the final numeric answer after '#### '."
CODE_SUFFIX = "Reply with only a Python code block."
QA_INSTRUCTION = "Answer with just the letter of the correct choice."
SUMM_INSTRUCTION = "Summarize the following dialogue in one or two sentences."

GSM8K_PROBE_ROWS = 300
SELECTION_SEED = 1234


def _load(name: str, split: str, config: str | None = None):
    return load_dataset(name, config, split=split, revision=REVISIONS[name])


def strip_calculator_markup(answer: str) -> str:
    """GSM8K rationales carry calculator annotations like '<<48/2=24>>'."""
    return re.sub(r"<<[^<>]*>>", "", answer)


def gsm8k_final(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def math_example(row: dict, uid: str) -> dict:
    return {
        "id": uid,
        "category": "math",
        "prompt": [{"role": "user", "content": f"{MATH_INSTRUCTION}\n\n{row['question']}"}],
        "target": strip_calculator_markup(row["answer"]),
        "reference": gsm8k_final(row["answer"]),
    }


def code_example(row: dict, uid: str) -> dict:
    text = row.get("prompt") or row["text"]
    tests = list(row["test_list"])
    content = f"{text}\n\nYour code must pass these tests:\n" + "\n".join(tests) + f"\n{CODE_SUFFIX}"
    return {
        "id": uid,
        "category": "code",
        "prompt": [{"role": "user", "content": content}],
        "target": f"```python\n{row['code'].strip()}\n```",
        "reference": {
            "tests": tests,
            "imports": list(row.get("test_imports") or []),
            "setup": row.get("test_setup_code") or "",
        },
    }


def qa_example(row: dict, uid: str) -> dict:
    labels = list(row["choices"]["label"])
    choices = "\n".join(f"{label}. {text}" for label, text in zip(labels, row["choices"]["text"]))
    return {
        "id": uid,
        "category": "qa",
        "prompt": [{"role": "user", "content": f"{row['question']}\n{choices}\n\n{QA_INSTRUCTION}"}],
        "target": row["answerKey"],
        "reference": {"answer": row["answerKey"], "labels": labels},
    }


def summ_example(row: dict, uid: str) -> dict:
    return {
        "id": uid,
        "category": "summ",
        "prompt": [{"role": "user", "content": f"{SUMM_INSTRUCTION}\n\n{row['dialogue']}"}],
        "target": row["summary"],
        "reference": row["summary"],
    }


def _rows(ds) -> list[dict]:
    return [dict(row) for row in ds]


def _sources(category: str, split: str) -> list[tuple[str, dict]]:
    """(stable id, raw row) pairs for one category and logical split."""
    if category == "math":
        if split == "eval":
            return [(f"gsm8k-test-{i}", r) for i, r in enumerate(_rows(_load("openai/gsm8k", "test", "main")))]
        rows = _rows(_load("openai/gsm8k", "train", "main"))
        cut = len(rows) - GSM8K_PROBE_ROWS
        chosen = range(cut, len(rows)) if split == "probe" else range(cut)
        return [(f"gsm8k-train-{i}", rows[i]) for i in chosen]
    if category == "code":
        if split == "eval":
            rows = _rows(_load("google-research-datasets/mbpp", "test", "sanitized"))
        elif split == "probe":
            rows = [*_rows(_load("google-research-datasets/mbpp", "validation", "full")),
                    *_rows(_load("google-research-datasets/mbpp", "prompt", "full"))]
        else:
            rows = _rows(_load("google-research-datasets/mbpp", "train", "full"))
        return [(f"mbpp-{r['task_id']}", r) for r in rows]
    if category == "qa":
        hf_split = {"train": "train", "probe": "validation", "eval": "test"}[split]
        rows = _rows(_load("allenai/ai2_arc", hf_split, "ARC-Challenge"))
        return [(f"arc-{r['id']}", r) for r in rows]
    if category == "summ":
        hf_split = {"train": "train", "probe": "validation", "eval": "test"}[split]
        rows = _rows(_load("knkarthick/samsum", hf_split))
        return [(f"samsum-{hf_split}-{r['id']}", r) for r in rows]
    raise ValueError(f"unknown category: {category}")


_BUILDERS: dict[str, Callable[[dict, str], dict]] = {
    "math": math_example, "code": code_example, "qa": qa_example, "summ": summ_example,
}


def select(items: list, n: int | None, seed: int = SELECTION_SEED) -> list:
    """Deterministic subset: a fixed shuffle, then the first n (all when n is None)."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    chosen = order if n is None else order[: min(n, len(items))]
    return [items[i] for i in sorted(chosen)]


def load_split(split: str, sizes: dict[str, int | None]) -> list[dict]:
    """split is "train", "probe" or "eval"; sizes maps category -> count (None = all)."""
    if split not in {"train", "probe", "eval"}:
        raise ValueError(f"unknown split: {split}")
    examples: list[dict] = []
    for category in CATEGORIES:
        if category not in sizes:
            continue
        pairs = select(_sources(category, split), sizes[category])
        examples.extend(_BUILDERS[category](row, uid) for uid, row in pairs)
    return examples
