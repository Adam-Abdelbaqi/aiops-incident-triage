import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import src.pipeline as pipeline_module


class DummyTextExtractor(BaseEstimator, TransformerMixin):
    """Fake text extractor for testing pipeline integration."""

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
        # Generate dummy embeddings with 32 features per row.
        rng = np.random.default_rng(42)
        return rng.normal(size=(len(X), 32))


def test_pipeline_fit_and_predict(monkeypatch):
    # Replace the real text extractor with our dummy.
    monkeypatch.setattr(
        pipeline_module,
        "TransformerExtractor",
        DummyTextExtractor,
    )

    n_rows = 50

    df = pd.DataFrame({
        "created_date": pd.date_range(
            "2026-01-01", periods=n_rows, freq="D"
        ),
        "subject": [f"Incident {i}" for i in range(n_rows)],
        "description": [
            f"Description of incident {i}"
            for i in range(n_rows)
        ],
        "sub_category": (
            ["Network", "Software", "Hardware", "Network", "Software"]
            * 10
        ),
        "department": ["IT", "Operations"] * 25,

        # This column must be excluded from the model.
        "product_service": ["Email", "Network"] * 25,

        "status": ["Open", "Resolved"] * 25,
        "category": ["Technical", "Service"] * 25,
        "satisfaction_rating": np.arange(n_rows),
        "resolved_date": pd.date_range(
            "2026-01-02", periods=n_rows, freq="D"
        ),
        "ticket_id": np.arange(n_rows),
        "resolution": ["Fixed", "Pending"] * 25,
        "priority": (
            ["Low", "Medium", "High", "Medium", "Low"] * 10
        ),
    })

    # Separate features from the target.
    X = df.drop(columns=["priority"])
    y = df["priority"]

    # Preserve chronological order.
    X_train, X_test = X.iloc[:40], X.iloc[40:]
    y_train = y.iloc[:40]

    # Build, train, and predict.
    pipeline = pipeline_module.build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # Verify the integration works.
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset(set(y_train.unique()))