"""Build a category-labeled prompt dataset from real benchmarks.

Categories: math (GSM8K), code (MBPP), qa (ARC-Challenge), summ (SamSum).
Each example: {"messages": [...], "category": str, "reference": str}.
"""

from datasets import Dataset, load_dataset

CATEGORIES = ["math", "code", "qa", "summ"]

ARC_INSTRUCTION = "Answer with just the letter of the correct choice."
MATH_INSTRUCTION = "Solve the problem. End with the final numeric answer after '#### '."
SUMM_INSTRUCTION = "Summarize the following dialogue in one or two sentences."


def _slice(ds, n: int, offset: int):
    end = min(offset + n, len(ds))
    if end - offset < n:
        print(f"warning: requested {n} examples at offset {offset}, split has {len(ds)}")
    return ds.select(range(offset, end))


def _math(split: str, n: int, offset: int) -> list[dict]:
    ds = _slice(load_dataset("openai/gsm8k", "main", split=split), n, offset)
    return [
        {
            "messages": [
                {"role": "user", "content": f"{MATH_INSTRUCTION}\n\n{ex['question']}"},
                {"role": "assistant", "content": ex["answer"]},
            ],
            "category": "math",
            "reference": ex["answer"].split("####")[-1].strip(),
        }
        for ex in ds
    ]


def _code(split: str, n: int, offset: int) -> list[dict]:
    ds = _slice(load_dataset("google-research-datasets/mbpp", "sanitized", split=split), n, offset)
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{ex['prompt']}\n\nYour code must pass these tests:\n"
                        + "\n".join(ex["test_list"])
                        + "\nReply with only a Python code block."
                    ),
                },
                {"role": "assistant", "content": f"```python\n{ex['code']}\n```"},
            ],
            "category": "code",
            "reference": "\n".join(ex["test_list"]),
        }
        for ex in ds
    ]


def _qa(split: str, n: int, offset: int) -> list[dict]:
    ds = _slice(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split), n, offset)
    out = []
    for ex in ds:
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])
        )
        out.append(
            {
                "messages": [
                    {"role": "user", "content": f"{ex['question']}\n{choices}\n\n{ARC_INSTRUCTION}"},
                    {"role": "assistant", "content": ex["answerKey"]},
                ],
                "category": "qa",
                "reference": ex["answerKey"],
            }
        )
    return out


def _summ(split: str, n: int, offset: int) -> list[dict]:
    ds = _slice(load_dataset("knkarthick/samsum", split=split), n, offset)
    return [
        {
            "messages": [
                {"role": "user", "content": f"{SUMM_INSTRUCTION}\n\n{ex['dialogue']}"},
                {"role": "assistant", "content": ex["summary"]},
            ],
            "category": "summ",
            "reference": ex["summary"],
        }
        for ex in ds
    ]


_LOADERS = {"math": _math, "code": _code, "qa": _qa, "summ": _summ}
_EVAL_SPLITS = {"math": "test", "code": "test", "qa": "test", "summ": "test"}


def build_dataset(
    n_per_category: int, split: str = "train", seed: int = 0, offset: int = 0
) -> Dataset:
    """offset keeps probe and eval sets disjoint when both draw from test splits."""
    rows: list[dict] = []
    for cat in CATEGORIES:
        actual_split = split if split == "train" else _EVAL_SPLITS[cat]
        rows.extend(_LOADERS[cat](actual_split, n_per_category, offset))
    return Dataset.from_list(rows).shuffle(seed=seed)
