import pandas as pd

def load_data(filepath: str , date_col:str = "created_date")-> pd.DataFrame:
    """
    Loads incident dataset from CSV and enforces chronological ordering.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file is not found: {filepath}") from e

    if df.empty:
        raise ValueError(f"The Dataset at {filepath} is empty")

    # Safety check: ensure the date column actually exists
    if date_col not in df.columns:
        raise KeyError(f"Expected date column '{date_col}' not found in dataset.")

    # Sort The Dataset Chornologically
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    print(f"The Dataset shape is {df.shape} with {df.shape[0]} Number of Rows and {df.shape[1]} Number of Columns")
    return df

def time_based_split(df: pd.DataFrame , train_size: float = 0.8):
    """Splits The Dataset Chornologically into Training And Testing Set"""
    if not 0.0 < train_size < 1.0:
        raise ValueError("The Training Size Must Be Between 0 And 1")

    split_idx = int(len(df) * train_size)
    training_df = df.iloc[:split_idx].copy().reset_index(drop= True)
    testing_df = df.iloc[split_idx:].copy().reset_index(drop= True)

    return training_df , testing_df