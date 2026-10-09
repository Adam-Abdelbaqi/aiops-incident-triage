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



def calculate_rolling_features(df: pd.DataFrame, date_col: str = "created_date") -> pd.DataFrame:
    """
    Pre-calculates stateful rolling features before the data enters the ML pipeline.
    Must be applied to the full historical dataset chronologically.
    """
    df_out = df.copy()
    
    # Ensure the column is datetime
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    
    # Extract just the date for grouping
    dates_only = df_out[date_col].dt.date
    
    # Calculate daily ticket volume
    daily_volume = df_out.groupby(dates_only).size()
    
    # Calculate trailing 3-day volume 
    # .shift(1) ensures today's prediction only uses the sum of the prior 3 days
    trailing_3d_volume = daily_volume.rolling(window=3, min_periods=1).sum().shift(1).fillna(0)
    
    # Map the pre-calculated volume back to every row in the dataframe
    df_out['global_volume_past_3d'] = dates_only.map(trailing_3d_volume)
    
    return df_out

def time_based_split(df: pd.DataFrame , train_size: float = 0.8)->tuple[pd.DataFrame, pd.DataFrame]:
    """Splits The Dataset Chronologically into Training And Testing Set"""

    if not 0.0 < train_size < 1.0:
        raise ValueError("The Training Size Must Be Between 0 And 1")

    split_idx = int(len(df) * train_size)
    training_df = df.iloc[:split_idx].copy().reset_index(drop= True)
    testing_df = df.iloc[split_idx:].copy().reset_index(drop= True)

    return training_df , testing_df