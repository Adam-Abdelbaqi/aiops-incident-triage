import pandas as pd
import pytest

from src.preprocessing.preprocess import check_missings, type_conversion


def test_check_missings():
    df = pd.DataFrame({
        "name": ["Adam", "Ali", None, "Omar"],
        "age": [20, None, 25, None]
    })

    result = check_missings(df)

    assert result["name"] == 25.0
    assert result["age"] == 50.0


def test_check_missings_invalid_input():
    with pytest.raises(TypeError):
        check_missings([1, 2, 3])


def test_check_missings_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        check_missings(df)


def test_type_conversion():
    df = pd.DataFrame({
        "sub_category": ["A", "B", "A"],
        "priority": ["High", "Low", "Medium"],
        "department": ["IT", "HR", "IT"],
        "subject": ["Server down", "Login issue", "Network issue"],
        "description": ["Server failure", "Cannot login", "Network failure"]
    })

    result = type_conversion(df)

    assert isinstance(result["sub_category"].dtype, pd.CategoricalDtype)
    assert isinstance(result["priority"].dtype, pd.CategoricalDtype)
    assert isinstance(result["department"].dtype, pd.CategoricalDtype)

    assert result["subject"].dtype == "string"
    assert result["description"].dtype == "string"


def test_type_conversion_does_not_modify_original():
    df = pd.DataFrame({
        "sub_category": ["A"],
        "priority": ["High"],
        "department": ["IT"],
        "subject": ["Server down"],
        "description": ["Server failure"]
    })

    original_dtypes = df.dtypes.copy()

    type_conversion(df)

    pd.testing.assert_series_equal(
        df.dtypes,
        original_dtypes
    )


def test_type_conversion_missing_columns():
    df = pd.DataFrame({
        "priority": ["High"],
        "subject": ["Server down"]
    })

    with pytest.raises(KeyError):
        type_conversion(df)


def test_type_conversion_invalid_input():
    with pytest.raises(TypeError):
        type_conversion(["not", "a", "dataframe"])


def test_type_conversion_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        type_conversion(df)