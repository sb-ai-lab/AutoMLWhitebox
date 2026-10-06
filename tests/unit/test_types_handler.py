import pandas as pd
import pytest

from autowoe.lib.types_handler.features_checkers_handlers import DEFAULT_DATE_FEATURE_TYPE
from autowoe.lib.types_handler.types_handler import TypesHandler


@pytest.mark.parametrize(
    "event_date",
    [
        pd.Series(pd.date_range("2020-01-01", periods=2), name="event_date"),
        pd.Series(["2020-01-01", "2020-01-02"], name="event_date"),
    ],
    ids=["datetime", "string"],
)
def test_date_feature_type_alias(event_date):
    train = pd.DataFrame({"event_date": event_date})

    transformed, public_types, private_types, _, _ = TypesHandler(
        train=train, public_features_type={"event_date": "date"}
    ).transform()

    expected_features = ["event_date__F__wd", "event_date__F__m", "event_date__F__y", "event_date__F__d"]
    assert public_types == {"event_date": DEFAULT_DATE_FEATURE_TYPE}
    assert list(private_types) == expected_features
    assert set(private_types.values()) == {"real"}
    assert list(transformed["event_date__F__wd"]) == [2, 3]


def test_unsupported_feature_type_error_is_actionable():
    train = pd.DataFrame({"feature": [1, 2]})

    with pytest.raises(ValueError, match="Use None .* 'date'.*date tuple"):
        TypesHandler(train=train, public_features_type={"feature": "unknown"}).transform()
