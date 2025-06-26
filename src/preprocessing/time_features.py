"""
time_features.py

Module for extracting time-based features from datetime columns.
"""

import pandas as pd


def extract_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Extracts hour, day, month, and weekday from a datetime column.

    Args:
        df (pd.DataFrame): Input DataFrame with a timestamp column.
        timestamp_col (str): Name of the timestamp column.

    Returns:
        pd.DataFrame: DataFrame with new time-based features.
    """
    df = df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Drop rows with invalid/missing timestamps
    df = df[df[timestamp_col].notna()]

    # Feature extraction
    df["transaction_hour"] = df[timestamp_col].dt.hour
    df["transaction_day"] = df[timestamp_col].dt.day
    df["transaction_month"] = df[timestamp_col].dt.month
    df["transaction_weekday"] = df[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6

    return df
