"""Teacher-selection policies. A policy maps one category-homogeneous batch to the
teacher(s) that score it.

fixed:    one named teacher for every batch
map:      a hand-written category -> teacher map
learned:  per category, the probe winner, but only when its paired-bootstrap CI against
          the default teacher excludes zero; otherwise the default teacher
random:   a uniformly random teacher per batch
ensemble: every teacher, logit-averaged
"""

import random

from mtkd.stats import paired_diff_ci

POLICIES = ("fixed", "map", "learned", "random", "ensemble")


def teachers_for_batch(condition: dict, category: str, teacher_keys: list[str],
                       rng: random.Random, learned_map: dict[str, str] | None = None) -> list[str]:
    policy = condition["policy"]
    if policy == "fixed":
        return [condition["teacher"]]
    if policy == "map":
        return [condition["map"][category]]
    if policy == "learned":
        if learned_map is None:
            raise ValueError("learned policy needs the probe's learned map (run the probe stage)")
        return [learned_map[category]]
    if policy == "random":
        return [rng.choice(teacher_keys)]
    if policy == "ensemble":
        return list(teacher_keys)
    raise ValueError(f"unknown policy: {policy}")


def select_learned_map(probe: dict[str, dict[str, list[float]]], default: str,
                       seed: int = 0) -> tuple[dict[str, str], dict]:
    """probe[teacher][category] holds per-item scores on the same items for every teacher.
    Returns the map and the evidence behind each choice."""
    categories = sorted(next(iter(probe.values())))
    chosen, evidence = {}, {}
    for category in categories:
        means = {t: sum(s[category]) / len(s[category]) for t, s in probe.items()}
        best = max(sorted(means), key=means.__getitem__)
        if best == default:
            chosen[category] = default
            evidence[category] = {"winner": best, "means": means, "vs_default": None}
            continue
        ci = paired_diff_ci(probe[best][category], probe[default][category], seed=seed)
        chosen[category] = best if ci["lo"] > 0 else default
        evidence[category] = {"winner": best, "means": means, "vs_default": ci}
    return chosen, evidence


def describe(condition: dict, teacher_keys: list[str], learned_map: dict[str, str] | None,
             categories: list[str]) -> dict[str, list[str]] | None:
    """The deterministic category -> teachers map a condition implies (None for random)."""
    if condition.get("kind") == "sft" or condition["policy"] == "random":
        return None
    rng = random.Random(0)
    return {c: teachers_for_batch(condition, c, teacher_keys, rng, learned_map) for c in categories}
