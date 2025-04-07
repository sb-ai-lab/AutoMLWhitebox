import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from autowoe import AutoWoE


def test_marked_values(train_data):
    df = train_data

    TARGET_NAME = "target"

    num_features = [col for col in df.columns if col.startswith("number")][:10]
    cat_features = [col for col in df.columns if col.startswith("string")][:5]

    df = df[num_features + cat_features + [TARGET_NAME]]

    df.iloc[:10, 0] = -1
    df.iloc[10:20, 0] = -2
    df.iloc[:20, 1] = 1234567890
    df.iloc[:20, 11] = "Special"

    train_df, test_df = train_test_split(df, stratify=df[TARGET_NAME], test_size=0.4, random_state=42, shuffle=True)

    assert all(train_df["string_1"].head(1) == "other")

    autowoe = AutoWoE(task="BIN", n_jobs=1, verbose=0)

    assert autowoe._params["l1_exp_scale"] == 4
    assert autowoe._params["imp_type"] == "feature_imp"
    assert autowoe._params["population_size"] is None
    assert not autowoe._params["monotonic"]

    none_params = (
        "woe_dict",
        "train_df",
        "split_dict",
        "target",
        "clf",
        "features_fit",
        "_cv_split",
        "_private_features_type",
        "_public_features_type",
        "_weights",
        "_intercept",
        "_p_vals",
        "feature_history",
    )
    for param in none_params:
        assert autowoe.__dict__[param] is None, f"This value should be None, but it's {autowoe.__dict__[param]}"

    start_fit_time = time.time()
    autowoe.fit(
        train=train_df,
        target_name=TARGET_NAME,
        features_mark_values={"number_0": (-1, -2), "number_1": (1234567890,), "string_1": ("Special",)},
    )

    assert time.time() - start_fit_time < 10, f"Fit time is {time.time() - start_fit_time}, it's more than 10"

    start_predict_time = time.time()
    test_pred = autowoe.predict_proba(test_df)
    assert time.time() - start_predict_time < 0.05, f"Diff is {time.time() - start_predict_time}, >= 0.05"

    score = roc_auc_score(test_df[TARGET_NAME], test_pred)

    assert score > 0.58

    assert autowoe.get_sql_inference_query("FEATURE_TABLE")

    representation = autowoe.get_model_represenation()

    features_representation = pd.DataFrame(representation["features"])

    assert all(
        np.isclose(features_representation["number_9"]["splits"], [7072.0, 11699.5, 13292.5])
    ), "There are different splits"

    assert np.isclose(representation["intercept"], -4.5482746), "There are different intercept coef"
