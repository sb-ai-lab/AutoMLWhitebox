import numpy as np
import pandas as pd

from autowoe.lib.cat_encoding.cat_encoding import CatEncoding
from autowoe.lib.pipelines.pipeline_feature_special_values import FeatureSpecialValues
from autowoe.lib.types_handler.features_checkers_handlers import cat_checker
from autowoe.lib.woe.woe import WoE


def test_cat_checker_supports_string_dtype():
    feature = pd.Series(["first", "second", None], dtype="str")

    assert cat_checker(feature)


def test_cat_checker_keeps_numeric_with_nan_real():
    feature = pd.Series([1, 2, 3, None])

    assert not cat_checker(feature)


def test_woe_supports_string_dtype():
    feature = pd.Series(["regular", "special"], dtype="str")
    woe = WoE(f_type="cat", split={"regular": 0})

    result = woe.split_feature(feature, {"special": None})

    assert result.tolist() == [0, "special"]


def test_woe_does_not_upcast_int_without_special_values():
    feature = pd.Series([1, 2], dtype="int64")
    woe = WoE(f_type="real", split=[1.5])

    result = woe.split_feature(feature, {"__NaN_0__": 0})

    assert result.tolist() == [0, 1]


def test_special_values_support_numeric_columns():
    processor = FeatureSpecialValues(mark_values={"feature": (-1,)})
    train, _ = processor.fit_transform(pd.DataFrame({"feature": [1.0, -1.0]}), {"feature": "real"})

    test, _ = processor.transform(pd.DataFrame({"feature": [-1.0, 2.0]}), ["feature"])

    assert train["feature"].tolist() == [1.0, "__Mark_0__"]
    assert test["feature"].tolist() == ["__Mark_0__", 2.0]


def test_unknown_numeric_category_becomes_small():
    processor = FeatureSpecialValues(th_cat=1, th_nan=1)
    processor.fit_transform(pd.DataFrame({"feature": [1, 2]}), {"feature": "cat"})

    test, _ = processor.transform(pd.DataFrame({"feature": [1, 3]}), ["feature"])

    assert test["feature"].tolist() == [1, "__Small_0__"]


def test_cat_encoding_replaces_integer_column_with_floats():
    encoder = CatEncoding(pd.DataFrame({"feature": [1, 1, 2, 2], "target": [0.0, 1.0, 0.0, 1.0]}))
    folds = {0: (np.array([0, 2]), np.array([1, 3])), 1: (np.array([1, 3]), np.array([0, 2]))}

    result = encoder(folds, np.array([], dtype=int))

    assert pd.api.types.is_float_dtype(result["feature"])


def test_special_values_fillna_nullable_int():
    processor = FeatureSpecialValues(th_nan=1, th_cat=1)
    train, spec = processor.fit_transform(
        pd.DataFrame({"feature": pd.array([1, 2, None, None], dtype="Int64")}), {"feature": "real"}
    )

    assert train["feature"].tolist()[:2] == [1, 2]
    assert train["feature"].tolist()[2:] == ["__NaN__", "__NaN__"]
    assert spec == {"feature": {"__NaN__": None}}


def test_special_values_transform_nullable_float():
    processor = FeatureSpecialValues(th_nan=100, th_cat=10)
    processor.fit_transform(
        pd.DataFrame({"feature": pd.array([1.0, 2.0, None] * 10, dtype="Float64")}), {"feature": "real"}
    )

    test, _ = processor.transform(pd.DataFrame({"feature": pd.array([1.0, None], dtype="Float64")}), ["feature"])

    assert test["feature"].tolist() == [1.0, "__NaN_0__"]


def test_special_values_mark_nullable_int():
    processor = FeatureSpecialValues(th_nan=2, th_cat=2, th_mark=2, mark_values={"feature": (-1,)})
    train, _ = processor.fit_transform(
        pd.DataFrame({"feature": pd.array([1, -1, 2, -1, None] * 2, dtype="Int64")}), {"feature": "real"}
    )

    assert train["feature"].tolist() == [1, "__Mark__-1__", 2, "__Mark__-1__", "__NaN__"] * 2


def test_special_values_categorical_dtype_with_marks():
    processor = FeatureSpecialValues(th_nan=1, th_cat=2, th_mark=10, mark_values={"feature": ("miss",)})
    train, _ = processor.fit_transform(
        pd.DataFrame({"feature": pd.Categorical(["a", "a", "b", "miss", None])}), {"feature": "cat"}
    )

    assert train["feature"].tolist() == ["a", "a", "__Small__", "__Mark_0__", "__NaN__"]


def test_special_values_transform_categorical_dtype():
    processor = FeatureSpecialValues(th_nan=1, th_cat=2)
    processor.fit_transform(pd.DataFrame({"feature": pd.Categorical(["a", "a", "b", None])}), {"feature": "cat"})

    test, _ = processor.transform(pd.DataFrame({"feature": pd.Categorical(["a", "c", None])}), ["feature"])

    assert test["feature"].tolist() == ["a", "__Small__", "__NaN__"]
