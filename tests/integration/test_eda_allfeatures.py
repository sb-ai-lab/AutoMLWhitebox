import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from pandas import Series

from autowoe import AutoWoE

from matplotlib import pyplot as plt

DATA_DIR = "examples/data/"


def test_eda_all_features():
    df = pd.read_csv(DATA_DIR + "train_demo.csv", low_memory=False)

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

    def plot_bar(woe_val):

        labels = woe_val["bin"].tolist()
        woe = woe_val["WoE"].tolist()
        freq = woe_val["count"].tolist()

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        _, ax = plt.subplots(figsize=[10, 5])
        ax.bar(x, freq, width, label="Score", hatch="///", edgecolor="#034569", color="none")

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = "#FF8B00"
        ax2.plot(
            x,
            woe,
            color=color,
            label="Time",
            marker="o",
        )
        ax2.set_ylabel("WoE", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.grid(False)
        diff = max(woe) - min(woe)
        ax2.set_ylim(np.min(woe) - 0.05 * diff, np.max(woe) + 0.05 * diff)

        ax.set_ylabel("Frequency")
        ax.set_title(woe_val.columns[0])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(np.min(freq) - 0.002, np.max(freq) + 0.002)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        plt.legend()
        plt.tight_layout()
        plt.show()

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
            mapper = {n: "({0}; {1}]".format(x, y) for (n, (x, y)) in enumerate(zip(["-inf"] + split, split + ["inf"]))}
            woe_val["bin"] = woe_val[col].map(mapper)
            woe_val["bin"] = np.where(woe_val["bin"].isnull().values, woe_val[col], woe_val["bin"])

        plot_bar(woe_val)

        if time.time() - start_time > 0.3:
            fails_counter += 1
    assert fails_counter <= 1, f"There were {fails_counter} fails, it's more than 1"
