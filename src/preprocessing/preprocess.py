import pandas as pd
from sklearn.preprocessing import LabelEncoder

def check_missings(df: pd.DataFrame) -> pd.Series:
    """
    Calculates missing value percentages for each column in the DataFrame.
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."
    assert not df.empty, "DataFrame is empty; cannot compute missing percentages."

    try:
        missing_percentages = (df.isna().sum() / len(df)) * 100
        return missing_percentages.round(2)
    except Exception as e:
        raise RuntimeError(f"Error calculating missing values: {e}") from e


def type_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts specified columns to appropriate categorical and string types.
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."
    assert not df.empty, "DataFrame is empty."

    df_clean = df.copy()
    categorical_cols = ["sub_category", "priority", "department"]
    text_cols = ["subject", "description"]

    # Validate column presence before type casting
    missing_cols = [col for col in categorical_cols + text_cols if col not in df_clean.columns]
    assert not missing_cols, f"The following expected columns are missing from the dataset: {missing_cols}"

    try:
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype("category")

        for col in text_cols:
            df_clean[col] = df_clean[col].astype("string")

        return df_clean

    except Exception as e:
        raise TypeError(f"Failed to convert column data types: {e}") from e


def encode_target_labels(df_train: pd.DataFrame, df_test: pd.DataFrame, target_col: str):
    """
    Encodes categorical target labels into numeric formats.
    Fits ONLY on the training data to prevent data leakage.
    """
    assert isinstance(df_train, pd.DataFrame), "df_train must be a pandas DataFrame."
    assert isinstance(df_test, pd.DataFrame), "df_test must be a pandas DataFrame."
    assert target_col in df_train.columns, f"Target column '{target_col}' missing from df_train."
    assert target_col in df_test.columns, f"Target column '{target_col}' missing from df_test."

    # Avoid SettingWithCopyWarning by working on explicit copies
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    encoder = LabelEncoder()

    try:
        # Fit on training set only
        train_copy['encoded_target'] = encoder.fit_transform(train_copy[target_col])
        
        # Check for target labels in test set that were not in train set
        unseen_classes = set(test_copy[target_col].unique()) - set(encoder.classes_)
        if unseen_classes:
            raise ValueError(f"Test set contains target labels unseen during training: {unseen_classes}")

        test_copy['encoded_target'] = encoder.transform(test_copy[target_col])

        return train_copy, test_copy, encoder

    except ValueError as ve:
        raise ValueError(f"Label encoding mismatch: {ve}") from ve
    except Exception as e:
        raise RuntimeError(f"Unexpected error during label encoding: {e}") from e