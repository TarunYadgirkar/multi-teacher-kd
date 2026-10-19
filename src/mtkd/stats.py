"""Bootstrap confidence intervals over items. Paired: both arms resample the same items."""

import numpy as np

N_BOOT = 10_000


def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def mean_ci(x, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    boots = x[_indices(len(x), n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def paired_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI of mean(a - b) over items, plus the bootstrap share of draws with diff <= 0."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boots = diff[_indices(len(diff), n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi),
            "p_le_0": float((boots <= 0).mean()), "n": int(len(diff))}


def normalized_gain(score, base):
    """Share of the remaining headroom recovered: (score - base) / (1 - base)."""
    return (score - base) / (1.0 - base)


def macro_gain_boot(scores: dict[str, np.ndarray], base: dict[str, np.ndarray],
                    n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Bootstrap draws of the macro (category-mean) normalized gain over base. Items are
    resampled within each category and shared between arms."""
    draws = []
    for k, category in enumerate(sorted(scores)):
        idx = _indices(len(scores[category]), n_boot, seed + k)
        s = np.asarray(scores[category], dtype=float)[idx].mean(axis=1)
        b = np.asarray(base[category], dtype=float)[idx].mean(axis=1)
        draws.append(normalized_gain(s, b))
    return np.mean(draws, axis=0)


def macro_gain(scores: dict[str, np.ndarray], base: dict[str, np.ndarray]) -> float:
    return float(np.mean([
        normalized_gain(np.mean(scores[c]), np.mean(base[c])) for c in sorted(scores)
    ]))


def summarize_draws(point: float, draws: np.ndarray, alpha: float = 0.05) -> dict:
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "p_le_0": float((draws <= 0).mean())}
