import numpy as np
import pytest

from mtkd.stats import macro_gain, macro_gain_boot, mean_ci, normalized_gain, paired_diff_ci


def test_mean_ci_brackets_mean():
    x = np.r_[np.ones(60), np.zeros(40)]
    ci = mean_ci(x)
    assert ci["mean"] == pytest.approx(0.6)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert 0.15 < ci["hi"] - ci["lo"] < 0.23


def test_paired_diff_detects_consistent_gain():
    base = np.r_[np.ones(100), np.zeros(100)]
    better = base.copy()
    better[100:130] = 1
    ci = paired_diff_ci(better, base)
    assert ci["mean"] == pytest.approx(0.15)
    assert ci["lo"] > 0 and ci["p_le_0"] == 0.0


def test_paired_diff_of_identical_arms_is_zero():
    x = np.random.default_rng(0).random(50)
    ci = paired_diff_ci(x, x)
    assert ci["mean"] == ci["lo"] == ci["hi"] == 0.0


def test_normalized_gain():
    assert normalized_gain(0.6, 0.5) == pytest.approx(0.2)
    assert normalized_gain(0.4, 0.5) == pytest.approx(-0.2)


def test_macro_gain_bootstrap_is_centred():
    rng = np.random.default_rng(0)
    base = {"a": (rng.random(400) < 0.4).astype(float), "b": rng.random(300) * 0.3}
    scores = {"a": np.maximum(base["a"], (rng.random(400) < 0.1)), "b": base["b"] + 0.05}
    point = macro_gain(scores, base)
    draws = macro_gain_boot(scores, base, n_boot=2000)
    assert np.quantile(draws, 0.025) < point < np.quantile(draws, 0.975)
    assert point > 0
