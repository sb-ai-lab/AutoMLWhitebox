import numpy as np
import pandas as pd
import pytest

from autowoe.lib.selectors import selector_first
from autowoe.lib.utilities.utils import TaskType


def test_nan_constant_selector_drops_columns():
    data = pd.DataFrame({"constant": [1, 1, 1], "variable": [1, 2, 3]})

    selected, feature_types = selector_first.nan_constant_selector(
        data, {"constant": "real", "variable": "real"}, th_const=0
    )

    assert list(selected.columns) == ["variable"]
    assert feature_types == {"variable": "real"}


def test_feature_imp_selector_drops_columns(monkeypatch):
    data = pd.DataFrame(
        {
            "important": np.arange(10),
            "unimportant": np.arange(10),
            "target": [0, 1] * 5,
        }
    )

    class Model:
        @staticmethod
        def feature_importance():
            return np.array([2, 1])

    monkeypatch.setattr(selector_first.lgb, "Dataset", lambda **kwargs: kwargs)
    monkeypatch.setattr(selector_first.lgb, "train", lambda **kwargs: Model())

    selected, feature_types = selector_first.feature_imp_selector(
        data=data,
        task=TaskType.BIN,
        features_type={"important": "real", "unimportant": "real"},
        features_mark_values=None,
        target_name="target",
        imp_th=0,
        imp_type="feature_imp",
        select_type=1,
        process_num=1,
    )

    assert list(selected.columns) == ["important", "target"]
    assert feature_types == {"important": "real"}


def test_permutation_selector_with_lightgbm():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=200)
    data = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=200),
            "target": (signal > 0).astype(int),
        }
    )

    selected, feature_types = selector_first.feature_imp_selector(
        data=data,
        task=TaskType.BIN,
        features_type={"signal": "real", "noise": "real"},
        features_mark_values=None,
        target_name="target",
        imp_th=0,
        imp_type="perm_imp",
        select_type=1,
        process_num=1,
    )

    assert list(selected.columns) == ["signal", "target"]
    assert feature_types == {"signal": "real"}


def test_feature_imp_selector_raises_on_empty_rows():
    data = pd.DataFrame({"f1": [-999.0] * 10, "target": [0, 1] * 5})

    with pytest.raises(ValueError, match="No rows left"):
        selector_first.feature_imp_selector(
            data=data,
            task=TaskType.BIN,
            features_type={"f1": "real"},
            features_mark_values={"f1": (-999,)},
            target_name="target",
            imp_th=0,
            imp_type="feature_imp",
            select_type=None,
            process_num=1,
        )


def test_feature_imp_selector_raises_on_empty_features():
    data = pd.DataFrame({"target": [0, 1] * 5})

    with pytest.raises(ValueError, match="No features left"):
        selector_first.feature_imp_selector(
            data=data,
            task=TaskType.BIN,
            features_type={},
            features_mark_values=None,
            target_name="target",
            imp_th=0,
            imp_type="feature_imp",
            select_type=None,
            process_num=1,
        )
