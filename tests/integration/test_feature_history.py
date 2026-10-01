import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import autowoe.lib.autowoe as autowoe_module
from autowoe import AutoWoE


@pytest.mark.parametrize(
    "prune_in_refit", [False, True], ids=["selector_only", "selector_and_refit"]
)
def test_refit_does_not_overwrite_selector_history(monkeypatch, prune_in_refit):
    selector_reason = "Low metric value"
    dropped = {}

    class Selector:
        def __init__(self, features_type, **kwargs):
            self.features = list(features_type)

        def __call__(self, feature_history, **kwargs):
            dropped["feature"] = self.features[-1]
            feature_history[dropped["feature"]] = selector_reason
            return self.features[:-1], None

    def model_fit(self, data_enc, features, valid_enc=None, valid_target=None):
        kept_features = features
        if prune_in_refit:
            dropped["refit_feature"] = features[0]
            kept_features = features[1:]
        features_fit = pd.Series(np.ones(len(kept_features)), index=kept_features)
        return {"features_fit": features_fit}, features_fit

    monkeypatch.setattr(autowoe_module, "Selector", Selector)
    monkeypatch.setattr(AutoWoE, "_model_fit", model_fit)

    features, target = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        random_state=42,
    )
    train = pd.DataFrame(features, columns=[f"feature_{idx}" for idx in range(features.shape[1])])
    train["target"] = target

    model = AutoWoE(n_jobs=1, debug=True, select_type=8)
    model.fit(train, target_name="target")

    assert model.feature_history[dropped["feature"]] == selector_reason
    if prune_in_refit:
        assert model.feature_history[dropped["refit_feature"]] == "Pruned during regression refit"
    assert all(model.feature_history[feature] is None for feature in model.features_fit.index)
