import warnings

import numpy as np
import pandas as pd
import pytest

from autowoe import AutoWoE
from autowoe.lib.utilities.refit import calc_p_val_reg


def _make_train(n=300, seed=0):
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    train["target"] = (train["f1"] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return train


def test_fit_raises_on_continuous_target_for_binary_task():
    train = _make_train()
    train["target"] = np.random.default_rng(1).normal(size=len(train))

    model = AutoWoE(task="BIN", n_jobs=1, verbose=0)

    with pytest.raises(ValueError, match="exactly 2 distinct values"):
        model.fit(train, target_name="target")


def test_fit_raises_when_all_features_filtered_by_nans():
    train = _make_train()
    train["f1"] = np.nan
    train["f2"] = np.nan

    model = AutoWoE(task="BIN", n_jobs=1, verbose=0)

    with pytest.raises(ValueError, match="No features left"):
        model.fit(train, target_name="target")


def test_fit_raises_when_all_features_pruned_by_selectors():
    train = _make_train()

    model = AutoWoE(task="BIN", metric_th=2.0, n_jobs=1, verbose=0)

    with pytest.raises(ValueError, match="All features were filtered out during selection"):
        model.fit(train, target_name="target")


def test_calc_p_val_reg_without_numpy_matrix():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 3))
    w = np.array([1.0, -2.0, 0.5])
    y = x @ w + 0.1 * rng.normal(size=200)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p_vals, b_var = calc_p_val_reg(x, y, w, 0.1)

    assert b_var is None
    assert p_vals.shape == (4,)
    assert np.all(np.isfinite(p_vals))
