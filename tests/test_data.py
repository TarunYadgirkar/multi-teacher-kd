import pytest

import mtkd.data as data


def test_strip_calculator_markup():
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72"
    assert data.strip_calculator_markup(answer) == "Natalia sold 48/2 = 24 clips.\n#### 72"


def test_math_example():
    ex = data.math_example({"question": "Q?", "answer": "a <<1+1=2>>2\n#### 1,234"}, "m0")
    assert ex["reference"] == "1234"
    assert "<<" not in ex["target"]
    assert ex["prompt"][0]["content"].endswith("Q?")


def test_code_example_keeps_test_imports_and_setup():
    row = {"prompt": "Write f.", "code": "def f(): pass", "test_list": ["assert f() is None"],
           "test_imports": ["import math"]}
    ex = data.code_example(row, "c0")
    assert ex["reference"] == {"tests": ["assert f() is None"], "imports": ["import math"], "setup": ""}
    assert "assert f() is None" in ex["prompt"][0]["content"]
    full_row = {"text": "Write g.", "code": "def g(): pass", "test_list": ["assert g() is None"],
                "test_setup_code": "X = 1"}
    assert data.code_example(full_row, "c1")["reference"]["setup"] == "X = 1"


def test_qa_example_carries_labels():
    row = {"question": "Which?", "choices": {"label": ["1", "2", "3"], "text": ["a", "b", "c"]},
           "answerKey": "2"}
    ex = data.qa_example(row, "q0")
    assert ex["reference"] == {"answer": "2", "labels": ["1", "2", "3"]}
    assert "2. b" in ex["prompt"][0]["content"]


def test_select_is_deterministic_and_order_preserving():
    items = list(range(100))
    assert data.select(items, 10) == data.select(items, 10)
    assert data.select(items, 10) == sorted(data.select(items, 10))
    assert data.select(items, None) == items
    assert data.select(items, 1000) == items


def _fake_load(name, split, config=None):
    if name == "openai/gsm8k":
        n = 400 if split == "train" else 50
        return [{"question": f"{split}{i}", "answer": "x\n#### 1"} for i in range(n)]
    if name == "google-research-datasets/mbpp":
        start = {"train": 601, "validation": 511, "prompt": 1, "test": 11}[split]
        return [{"task_id": start + i, "text": "t", "prompt": "t", "code": "c", "test_list": ["assert 1"],
                 "test_imports": [], "test_setup_code": ""} for i in range(10)]
    if name == "allenai/ai2_arc":
        return [{"id": f"{split}-{i}", "question": "q", "answerKey": "A",
                 "choices": {"label": ["A", "B"], "text": ["x", "y"]}} for i in range(30)]
    if name == "knkarthick/samsum":
        return [{"id": str(i), "dialogue": "d", "summary": "s"} for i in range(30)]
    raise KeyError(name)


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setattr(data, "_load", _fake_load)


def test_splits_are_disjoint(fake_hub):
    sizes = {c: None for c in data.CATEGORIES}
    ids = {split: {ex["id"] for ex in data.load_split(split, sizes)} for split in ("train", "probe", "eval")}
    assert not ids["train"] & ids["probe"]
    assert not ids["train"] & ids["eval"]
    assert not ids["probe"] & ids["eval"]


def test_gsm8k_probe_is_carved_out_of_train(fake_hub):
    probe = data.load_split("probe", {"math": None})
    train = data.load_split("train", {"math": None})
    assert len(probe) == data.GSM8K_PROBE_ROWS
    assert len(train) == 400 - data.GSM8K_PROBE_ROWS


def test_unknown_split_rejected(fake_hub):
    with pytest.raises(ValueError):
        data.load_split("test", {"math": 1})
