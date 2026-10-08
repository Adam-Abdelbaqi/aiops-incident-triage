import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError

from src.models.classifier import RandomForestModel
from src.features.dim_reducer import EmbeddingReduction


def test_random_forest_fit():
    X = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6],
        "feature_2": [6, 5, 4, 3, 2, 1]})

    y = pd.Series([
        "Low", "Low", "Medium",
        "Medium", "High", "High"])

    model = RandomForestModel(n_estimators=10, random_state=42)

    result = model.fit(X, y)

    assert result is model
    assert model.is_fitted_
    assert hasattr(model, "model_")
    assert model.feature_names_ == ["feature_1", "feature_2"]


def test_random_forest_predict():
    X_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4, 5, 6],
        "feature_2": [6, 5, 4, 3, 2, 1]})

    y_train = pd.Series([
        "Low", "Low", "Medium",
        "Medium", "High", "High"])

    X_test = pd.DataFrame({
        "feature_1": [1.5, 3.5, 5.5],
        "feature_2": [5.5, 3.5, 1.5]})

    model = RandomForestModel(n_estimators=10, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)


def test_random_forest_predict_before_fit():
    X_test = pd.DataFrame({
        "feature_1": [1, 2],
        "feature_2": [2, 1]})

    model = RandomForestModel()

    with pytest.raises(RuntimeError):
        model.predict(X_test)


def test_random_forest_feature_mismatch():
    X_train = pd.DataFrame({
        "feature_1": [1, 2, 3, 4],
        "feature_2": [4, 3, 2, 1]})

    y_train = pd.Series([
        "Low", "Low", "High", "High"])

    X_test = pd.DataFrame({
        "feature_1": [1, 2, 3]})

    model = RandomForestModel(n_estimators=10, random_state=42)

    model.fit(X_train, y_train)

    with pytest.raises(ValueError):
        model.predict(X_test)


def test_random_forest_numpy_input():
    X = np.array([
        [1, 6],
        [2, 5],
        [3, 4],
        [4, 3],
        [5, 2],
        [6, 1]])

    y = pd.Series([
        "Low", "Low", "Medium",
        "Medium", "High", "High"])

    model = RandomForestModel(n_estimators=10, random_state=42)

    model.fit(X, y)

    predictions = model.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert model.feature_names_ == ["component_1", "component_2"]


def test_embedding_reduction_fit():
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
        [5.0, 10.0, 15.0]])

    reducer = EmbeddingReduction(n_components=2, random_state=42)

    result = reducer.fit(X)

    assert result is reducer
    assert hasattr(reducer, "pca_")


def test_embedding_reduction_transform():
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
        [5.0, 10.0, 15.0]])

    reducer = EmbeddingReduction(n_components=2, random_state=42)

    reducer.fit(X)

    X_reduced = reducer.transform(X)

    assert isinstance(X_reduced, np.ndarray)
    assert X_reduced.shape == (5, 2)


def test_embedding_reduction_transform_before_fit():
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0]])

    reducer = EmbeddingReduction(n_components=2)

    with pytest.raises(NotFittedError):
        reducer.transform(X)
