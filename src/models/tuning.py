import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def validate_tuning_data(X_train: pd.DataFrame,y_train: pd.Series):

    # Validate types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a Pandas DataFrame")

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a Pandas Series")

    # Validate non-empty data
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("X_train and y_train cannot be empty")

    # Validate matching number of samples
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} samples "
            f"but y_train has {len(y_train)} samples")



def tune_hyperparameters(X_train:pd.DataFrame , y_train:pd.Series):

    # Validate training data
    validate_tuning_data(X_train , y_train)

    # Base Random Forest
    rf = RandomForestClassifier(random_state=42)

    # Hyperparameter search space
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "max_features": [0.6, 0.8],
        "min_samples_split": [2, 8, 10],
        "class_weight": [None, "balanced"]
    }

    # Randomized hyperparameter search
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    # Run the search
    random_search.fit(X_train, y_train)

    return random_search.best_params_
   