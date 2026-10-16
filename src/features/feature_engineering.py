import pandas as pd
from sklearn.base import TransformerMixin , BaseEstimator

class TabularExtractor(BaseEstimator , TransformerMixin):
    def __init__(self , categorical_features = ('category' , 'sub_category'), date_col = "created_date"):
        """Select the categorical features used by the tabular branch."""

        self.categorical_features = categorical_features

        self.date_col = date_col
        self.sub_category_map_ = {}
        self.department_map_ = {}

    def fit(self , X: pd.DataFrame , y = None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabularExtractor expects a pandas DataFrame.")

        missing_columns = [col for col in self.categorical_features if col not in X.columns]

        if missing_columns:
            raise ValueError(f"Missing required features: {missing_columns}")

        if "priority" in X.columns:
            raise ValueError("Remove the target column 'priority' from X before fitting.")

        return self

    def transform(self , X:pd.DataFrame):
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TabularExtractor expects a pandas DataFrame.")

        if "priority" in X.columns:
            raise ValueError(
                "The target column 'priority' must not be in X."
            )

        return X.loc[:, list(self.categorical_features)].copy()

    