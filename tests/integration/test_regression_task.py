import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from autowoe import AutoWoE

DATA_DIR = "examples/data/"


def test_regression_task(regression_data):

    df = regression_data

    TARGET_NAME = "Target"

    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)

    autowoe = AutoWoE(
        task="REG",
        monotonic=True,
        interpreted_model=True,
        regularized_refit=True,
        metric_th=0.0,
        n_jobs=1,
        verbose=0,
    )

    start_fit_time = time.time()
    autowoe.fit(train=train_df, target_name=TARGET_NAME)

    assert time.time() - start_fit_time < 25

    start_predicts_time = time.time()

    train_pred = autowoe.predict(train_df)
    test_pred = autowoe.predict(test_df)

    train_pred = autowoe.predict(train_df)

    assert time.time() - start_predicts_time < 0.3, f"Pred time is {time.time() - start_predicts_time}, >= 0.3"

    r2_train = r2_score(train_df[TARGET_NAME], train_pred)
    r2_test = r2_score(test_df[TARGET_NAME], test_pred)

    assert r2_train > 0.8
    assert r2_test > 0.76

    autowoe.get_sql_inference_query("FEATURE_TABLE")
