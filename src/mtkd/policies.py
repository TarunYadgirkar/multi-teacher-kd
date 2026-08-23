"""Teacher-selection policies. Each returns teacher index per example.

fixed:    always teachers[0]
random:   uniform over teachers (seeded)
strongest: single teacher with best overall probe score
task:     hand-specified category -> teacher map from config
learned:  per-category argmax over probe win-rates
"""

import json
import random


def assign_teachers(
    dataset,
    policy: str,
    teachers: list[str],
    category_map: dict[str, str] | None = None,
    probe_results_path: str | None = None,
    seed: int = 0,
):
    n = len(teachers)
    if policy == "fixed":
        ids = [0] * len(dataset)
    elif policy == "random":
        rng = random.Random(seed)
        ids = [rng.randrange(n) for _ in range(len(dataset))]
    elif policy == "strongest":
        probe = _load_probe(probe_results_path)
        best = max(probe, key=lambda t: probe[t]["overall"])
        ids = [teachers.index(best)] * len(dataset)
    elif policy == "task":
        if not category_map:
            raise ValueError("task policy requires category_map in config")
        ids = [teachers.index(category_map[ex["category"]]) for ex in dataset]
    elif policy == "learned":
        probe = _load_probe(probe_results_path)
        by_cat = {
            cat: max(probe, key=lambda t: probe[t].get(cat, 0.0))
            for cat in {ex["category"] for ex in dataset}
        }
        ids = [teachers.index(by_cat[ex["category"]]) for ex in dataset]
    else:
        raise ValueError(f"unknown policy: {policy}")
    return dataset.add_column("teacher_id", ids)


def _load_probe(path: str | None) -> dict:
    if path is None:
        raise ValueError("this policy requires probe_results_path (run mtkd.probe first)")
    with open(path) as f:
        return json.load(f)
