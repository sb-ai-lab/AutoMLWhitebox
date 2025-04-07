import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pandas import Series

from autowoe import AutoWoE


def test_eda_all_features(train_data):
    df = train_data

    TARGET_NAME = "target"

    num_features = [col for col in df.columns if col.startswith("number")][:10]
    cat_features = [col for col in df.columns if col.startswith("string")][:5]

    df = df[num_features + cat_features + [TARGET_NAME]]

    train_df, test_df = train_test_split(df, stratify=df[TARGET_NAME], test_size=0.4, random_state=42, shuffle=True)

    autowoe = AutoWoE(
        task="BIN",
        n_jobs=1,
        verbose=0,
        # turn off initial importance selection - this step force all features to pass into the binning stage
        imp_th=-1,
    )

    autowoe.fit(train=train_df, target_name=TARGET_NAME)

    test_pred = autowoe.predict_proba(test_df)

    score = roc_auc_score(test_df[TARGET_NAME], test_pred)

    assert np.isclose(score, 0.6186, atol=1e-4), f"Real score is {score}"

    enc = autowoe.test_encoding(train_df, list(autowoe.woe_dict.keys()), bins=True)
    fails_counter = 0
    for col in enc.columns:
        start_time = time.time()

        grp = enc.groupby(col).size()
        woe = autowoe.woe_dict[col]

        woe_val = Series(woe.cod_dict).reset_index()
        woe_val.columns = [col, "WoE"]
        woe_val["count"] = woe_val[col].map(grp).fillna(0).values.astype(int)
        if woe.f_type == "cat":
            woe_val["bin"] = woe_val[col]
        else:
            split = list(woe.split.astype(np.float32))
            mapper = {n: f"({x}; {y}]" for (n, (x, y)) in enumerate(zip(["-inf"] + split, split + ["inf"]))}
            woe_val["bin"] = woe_val[col].map(mapper)
            woe_val["bin"] = np.where(woe_val["bin"].isnull().values, woe_val[col], woe_val["bin"])

        if time.time() - start_time > 0.3:
            fails_counter += 1
    assert fails_counter <= 1, f"There were {fails_counter} fails, it's more than 1"
