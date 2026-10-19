import random

import pytest

from mtkd.policies import describe, select_learned_map, teachers_for_batch

KEYS = ["instruct", "coder", "math"]


def test_fixed_and_map():
    rng = random.Random(0)
    assert teachers_for_batch({"policy": "fixed", "teacher": "coder"}, "qa", KEYS, rng) == ["coder"]
    spec = {"policy": "map", "map": {"math": "math", "qa": "instruct"}}
    assert teachers_for_batch(spec, "math", KEYS, rng) == ["math"]


def test_random_is_seeded_and_covers_teachers():
    picks = [teachers_for_batch({"policy": "random"}, "qa", KEYS, random.Random(5))[0] for _ in range(3)]
    assert len(set(picks)) == 1
    rng = random.Random(0)
    assert {teachers_for_batch({"policy": "random"}, "qa", KEYS, rng)[0] for _ in range(60)} == set(KEYS)


def test_ensemble_uses_every_teacher():
    assert teachers_for_batch({"policy": "ensemble"}, "code", KEYS, random.Random(0)) == KEYS


def test_learned_requires_map():
    with pytest.raises(ValueError):
        teachers_for_batch({"policy": "learned"}, "qa", KEYS, random.Random(0))


def test_learned_map_needs_a_separated_ci():
    n = 300
    probe = {
        "instruct": {"code": [1.0] * 150 + [0.0] * 150, "math": [1.0] * 150 + [0.0] * 150},
        "coder": {"code": [1.0] * 280 + [0.0] * 20, "math": [1.0] * 150 + [0.0] * 150},
        "math": {"code": [0.0] * n, "math": [1.0] * 152 + [0.0] * 148},
    }
    chosen, evidence = select_learned_map(probe, "instruct")
    assert chosen == {"code": "coder", "math": "instruct"}
    assert evidence["math"]["winner"] == "math"
    assert evidence["math"]["vs_default"]["lo"] <= 0


def test_describe():
    spec = {"policy": "map", "map": {"math": "math", "qa": "instruct"}}
    assert describe(spec, KEYS, None, ["math", "qa"]) == {"math": ["math"], "qa": ["instruct"]}
    assert describe({"policy": "random"}, KEYS, None, ["qa"]) is None
    assert describe({"kind": "sft"}, KEYS, None, ["qa"]) is None
