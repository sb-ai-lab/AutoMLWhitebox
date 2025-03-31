#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pytest


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


DATA_DIR = "examples/data/"


@pytest.fixture()
def train_data():
    train = pd.read_csv(
        DATA_DIR + "train_demo.csv",
        low_memory=False,
        index_col="line_id",
        parse_dates=["datetime_" + str(i) for i in range(2)],
    )
    return train


@pytest.fixture()
def test_data():
    test = pd.read_csv(
        DATA_DIR + "test_demo.csv", index_col="line_id", parse_dates=["datetime_" + str(i) for i in range(2)]
    )
    return test


@pytest.fixture()
def test_target():
    test_target = pd.read_csv(DATA_DIR + "test-target_demo.csv")["target"]
    return test_target

@pytest.fixture()
def cat_data():
    data = pd.read_csv(DATA_DIR + "data_cat.csv")
    return data

@pytest.fixture()
def regression_data():
    data = pd.read_csv(DATA_DIR + "regression_dataset.csv")
    return data
