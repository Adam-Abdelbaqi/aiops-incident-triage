import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.pipeline as pipeline_module


class DummyTextExtractor(BaseEstimator , TransformerMixin):
    """Lightweight replacement for the real text embedding model."""

    def __init__(
        self,
        text_columns=("subject", "description"),
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    ):
        self.text_columns = text_columns
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for _, row in X.iterrows():
            subject = str(row["subject"])
            description = str(row["description"])

            features.append([
                len(subject),
                len(description),
            ])

        return np.asarray(features, dtype=np.float32)


@pytest.fixture
def sample_data():
    """Create sample incident data for testing."""

    X = pd.DataFrame({
        "subject": [
            "login issue",
            "slow application",
            "server outage",
            "password reset",
            "database error",
            "network failure",
            "display issue",
            "service outage",
            "account locked",
            "slow database",
            "server error",
            "network issue",
        ],
        "description": [
            "cannot login",
            "application is slow",
            "production server down",
            "reset user password",
            "database connection failed",
            "network is unavailable",
            "display formatting problem",
            "service is unavailable",
            "account access failure",
            "database response is slow",
            "server is not responding",
            "network connection failed",
        ],
        "category": [
            "Access", "Performance", "Infrastructure", "Access",
            "Infrastructure", "Infrastructure", "Application",
            "Infrastructure", "Access", "Performance",
            "Infrastructure", "Infrastructure",
        ],
        "sub_category": [
            "Login", "Latency", "Server", "Password",
            "Database", "Network", "UI", "Server",
            "Login", "Database", "Server", "Network",
        ],
    })

    y = pd.Series([
        "Low", "Medium", "Critical", "Low",
        "High", "Critical", "Low", "High",
        "Medium", "Medium", "Critical", "High",
    ])

    return X, y


@pytest.fixture
def model_pipeline(monkeypatch):
    """Build the pipeline with its production configuration."""

    monkeypatch.setattr(
        pipeline_module,
        "TransformerExtractor",
        DummyTextExtractor,
    )

    return pipeline_module.build_pipeline(C=100.0)


@pytest.fixture
def fittable_pipeline(monkeypatch):
    """Build a pipeline with more iterations for test-data fitting."""

    monkeypatch.setattr(
        pipeline_module,
        "TransformerExtractor",
        DummyTextExtractor,
    )

    pipeline = pipeline_module.build_pipeline(C=100.0)
    pipeline.named_steps["classifier"].set_params(max_iter=10000)

    return pipeline


def test_build_pipeline_returns_pipeline(model_pipeline):
    assert isinstance(model_pipeline, Pipeline)

    assert "features" in model_pipeline.named_steps
    assert "classifier" in model_pipeline.named_steps


def test_pipeline_uses_logistic_regression(model_pipeline):
    classifier = model_pipeline.named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == 100.0
    assert classifier.max_iter == 2000
    assert classifier.random_state == 42


def test_pipeline_predict_before_fit_raises(
    model_pipeline,
    sample_data,
):
    X, _ = sample_data

    with pytest.raises(NotFittedError):
        model_pipeline.predict(X)


def test_pipeline_fit_returns_self(fittable_pipeline, sample_data):
    X, y = sample_data

    result = fittable_pipeline.fit(X, y)

    assert result is fittable_pipeline


def test_pipeline_predict_returns_valid_output(
    fittable_pipeline,
    sample_data,
):
    X, y = sample_data

    fittable_pipeline.fit(X, y)
    predictions = fittable_pipeline.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X),)
    assert set(predictions).issubset(set(y))


def test_pipeline_handles_unseen_categories(
    fittable_pipeline,
    sample_data,
):
    X_train, y_train = sample_data

    fittable_pipeline.fit(X_train, y_train)

    X_new = pd.DataFrame({
        "subject": ["new issue"],
        "description": ["a previously unseen incident"],
        "category": ["UnseenCategory"],
        "sub_category": ["UnseenSubcategory"],
    })

    predictions = fittable_pipeline.predict(X_new)

    assert predictions.shape == (1,)
    assert predictions[0] in fittable_pipeline.classes_
