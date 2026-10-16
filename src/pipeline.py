from src.features.feature_engineering import TabularExtractor
from src.features.feature_extraction import TransformerExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np



def tabular_to_numpy(X: pd.DataFrame) -> np.ndarray:
    """Return only numeric tabular features as a NumPy array."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("TabularExtractor must return a pandas DataFrame.")

    # These columns are processed by the text branch, not the tabular branch.
    text_columns = ["subject", "description"]
    X = X.drop(columns=[col for col in text_columns if col in X.columns])

    if "priority" in X.columns:
        raise ValueError("The target column 'priority' must be removed from X before fitting.")

    non_numeric_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_columns:
        raise TypeError(
            "Non-numeric columns remain in the tabular features: "
            f"{non_numeric_columns}. Check the columns dropped by TabularExtractor.")

    if X.isna().any().any():
        missing_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values remain in the tabular features: {missing_columns}")

    return X.to_numpy(dtype=np.float32)


def build_pipeline(C = 100.0):
    """Combine text features, tabular features, and the classifier."""

    text_pipeline = Pipeline(
        steps=[
            (
                "text_extractor",
                TransformerExtractor(
                    text_columns=("subject", "description"),
                    model_name="all-MiniLM-L6-v2",
                    batch_size=32,
                ),
            ),
            
        ])

    # The text columns are removed here because they are handled by the text branch.
    # Keep product_service out of X as decided during feature selection.
    tabular_pipeline = Pipeline(
        steps=[
            (
                "tabular_extractor",
                TabularExtractor(categorical_features=('category' , 'sub_category')),
            ),

            ("Encoder" , OneHotEncoder(handle_unknown= "ignore")),

        
        ])

    combined_features = FeatureUnion(
        transformer_list=[
            ("text_features", text_pipeline),
            ("tabular_features", tabular_pipeline),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("features", combined_features),
            ("classifier", LogisticRegression(C= C, class_weight= None, max_iter= 2000, random_state= 42)),
        ]
    )

    return pipeline
