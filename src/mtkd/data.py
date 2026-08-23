"""Build a category-labeled prompt dataset from real benchmarks.

Categories: math (GSM8K), code (MBPP), qa (ARC-Challenge), summ (SamSum).
Each example: {"messages": [...], "category": str, "reference": str}.
"""

from datasets import Dataset, load_dataset

CATEGORIES = ["math", "code", "qa", "summ"]

ARC_INSTRUCTION = "Answer with just the letter of the correct choice."
MATH_INSTRUCTION = "Solve the problem. End with the final numeric answer after '#### '."
SUMM_INSTRUCTION = "Summarize the following dialogue in one or two sentences."


def _math(split: str, n: int) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split).select(range(n))
    return [
        {
            "messages": [{"role": "user", "content": f"{MATH_INSTRUCTION}\n\n{ex['question']}"}],
            "category": "math",
            "reference": ex["answer"].split("####")[-1].strip(),
        }
        for ex in ds
    ]


def _code(split: str, n: int) -> list[dict]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split).select(range(n))
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
                }
            ],
            "category": "code",
            "reference": "\n".join(ex["test_list"]),
        }
        for ex in ds
    ]


def _qa(split: str, n: int) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split).select(range(n))
    out = []
    for ex in ds:
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])
        )
        out.append(
            {
                "messages": [
                    {"role": "user", "content": f"{ex['question']}\n{choices}\n\n{ARC_INSTRUCTION}"}
                ],
                "category": "qa",
                "reference": ex["answerKey"],
            }
        )
    return out


def _summ(split: str, n: int) -> list[dict]:
    ds = load_dataset("knkarthick/samsum", split=split).select(range(n))
    return [
        {
            "messages": [{"role": "user", "content": f"{SUMM_INSTRUCTION}\n\n{ex['dialogue']}"}],
            "category": "summ",
            "reference": ex["summary"],
        }
        for ex in ds
    ]


_LOADERS = {"math": _math, "code": _code, "qa": _qa, "summ": _summ}
_EVAL_SPLITS = {"math": "test", "code": "test", "qa": "test", "summ": "test"}


def build_dataset(n_per_category: int, split: str = "train", seed: int = 0) -> Dataset:
    rows: list[dict] = []
    for cat in CATEGORIES:
        actual_split = split if split == "train" else _EVAL_SPLITS[cat]
        rows.extend(_LOADERS[cat](actual_split, n_per_category))
    return Dataset.from_list(rows).shuffle(seed=seed)
