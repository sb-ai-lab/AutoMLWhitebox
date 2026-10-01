import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    assert len(enc) == len(train_df)
    assert set(enc.columns) == set(autowoe.woe_dict)
