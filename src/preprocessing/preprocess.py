import pandas as pd
from sklearn.preprocessing import LabelEncoder

def check_missings(df: pd.DataFrame) -> pd.Series:
    """
    Calculates missing value percentages for each column in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame): 
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is Empty; cannot compute missing percentages.")

    try:
        missing_percentages = (df.isna().sum() / len(df)) * 100
        return missing_percentages.round(2)
    except Exception as e:
        raise ValueError(f"Error calculating missing values: {e}") from e


def type_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts specified columns to appropriate categorical and string types.
    """
    if not isinstance(df, pd.DataFrame): 
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is Empty.")


    df_clean = df.copy()
    categorical_cols = ["sub_category", "priority", "department"]
    text_cols = ["subject", "description"]

    # Validate column presence before type casting
    missing_cols = [col for col in categorical_cols + text_cols if col not in df_clean.columns]
    if missing_cols:
        raise KeyError(f"The following expected columns are missing from the dataset: {missing_cols}")

    try:
        for col in categorical_cols:
            df_clean[col] = df_clean[col].astype("category")

        for col in text_cols:
            df_clean[col] = df_clean[col].astype("string")

        return df_clean

    except Exception as e:
        raise TypeError(f"Failed to convert column data types: {e}") from e

