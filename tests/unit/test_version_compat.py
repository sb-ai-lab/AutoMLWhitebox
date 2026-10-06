import numpy as np
import pandas as pd
import pytest

from autowoe.lib.optimizer import optimizer
from autowoe.lib.selectors import selector_first
from autowoe.lib.utilities import refit
from autowoe.lib.utilities.utils import TaskType


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.1.99", False),
        ("1.2.0", True),
        ("1.10.0", True),
        ("1.2.0rc1", False),
        ("1.2.dev0", False),
        ("1.2.0+cpu", True),
    ],
)
def test_sklearn_version_comparison_uses_pep440_ordering(monkeypatch, version, expected):
    monkeypatch.setattr(refit.sklearn, "__version__", version)

    assert refit._sklearn_at_least(1, 2) is expected


@pytest.mark.parametrize(
    ("version", "expected_penalty"),
    [
        ("1.1.99", "none"),
        ("1.2.0", None),
        ("1.10.0", None),
        ("1.2.0rc1", "none"),
        ("1.2.dev0", "none"),
        ("1.2.0+cpu", None),
    ],
)
def test_logreg_penalty_uses_pep440_compatibility_policy(monkeypatch, version, expected_penalty):
    monkeypatch.setattr(refit.sklearn, "__version__", version)

    assert refit._logreg_penalty() == expected_penalty


@pytest.mark.parametrize(
    ("version", "expected_penalty"), [("1.1.99", "none"), ("1.10.0", None)])
def test_validation_refit_passes_compatible_penalty_to_sklearn(monkeypatch, version, expected_penalty):
    init_kwargs = {}

    class LogisticRegressionStub:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)

        def fit(self, x_train, y):
            self.coef_ = np.array([[1.0]])
            self.intercept_ = np.array([0.0])

    monkeypatch.setattr(refit.sklearn, "__version__", version)
    monkeypatch.setattr(refit, "LogisticRegression", LogisticRegressionStub)
    monkeypatch.setattr(refit, "calc_p_val", lambda *_: (np.array([0.5]), np.array([1.0])))

    refit.calc_p_val_on_valid(np.array([[0.0], [1.0]]), np.array([0, 1]), TaskType.BIN)

    assert init_kwargs["penalty"] == expected_penalty


def test_feature_importance_selector_uses_modern_lightgbm_callbacks(monkeypatch):
    data = pd.DataFrame({"important": range(10), "unimportant": range(10), "target": [0, 1] * 5})
    train_kwargs = {}

    class Model:
        @staticmethod
        def feature_importance():
            return np.array([2, 1])

    monkeypatch.setattr(selector_first.lgb, "Dataset", lambda **kwargs: kwargs)
    monkeypatch.setattr(selector_first.lgb, "log_evaluation", lambda period: ("log", period))
    monkeypatch.setattr(
        selector_first.lgb, "early_stopping", lambda stopping_rounds, first_metric_only, verbose: ("stop", stopping_rounds)
    )
    monkeypatch.setattr(selector_first.lgb, "train", lambda **kwargs: train_kwargs.update(kwargs) or Model())

    selector_first.feature_imp_selector(
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

    assert train_kwargs["callbacks"] == [("log", selector_first.verbose_eval), ("stop", 10)]


def test_tree_optimizer_uses_modern_lightgbm_cv_result_key(monkeypatch):
    optimizer_ = optimizer.TreeParamOptimizer.__new__(optimizer.TreeParamOptimizer)
    optimizer_._task = TaskType.BIN
    optimizer_._metric = "auc"
    optimizer_._lgb_train = object()
    monkeypatch.setattr(
        optimizer.TreeParamOptimizer,
        "_TreeParamOptimizer__get_folds",
        lambda self, seed: [(np.array([0]), np.array([1]))],
    )
    monkeypatch.setattr(
        optimizer.lgb,
        "cv",
        lambda **kwargs: {"valid auc-mean": [0.75]},
    )

    scores = optimizer_._TreeParamOptimizer__get_scores({}, n=2)

    assert scores == [[0.75], [0.75]]
