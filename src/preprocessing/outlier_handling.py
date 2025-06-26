"""
outlier_handling.py

This module provides utilities for detecting and handling outliers
and applying transformations to skewed numerical features.
"""

import numpy as np
import pandas as pd
from typing import Optional


def log_transform(
    df: pd.DataFrame, column: str, new_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Applies a log1p transformation to reduce skewness in a numerical feature.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): Column to transform.
        new_column (str, optional): Name for the transformed column.
                                    If None, appends '_log' to original name.

    Returns:
        pd.DataFrame: DataFrame with new transformed column.
    """
    df = df.copy()
    col_out = new_column or f"{column}_log"
    df[col_out] = df[column].apply(
        lambda x: np.log1p(abs(x))
    )  # use abs to handle negatives
    return df


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    lower_quantile=0.01,
    upper_quantile=0.99,
    new_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Caps outliers using specified lower and upper quantile thresholds.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): Column to cap.
        lower_quantile (float): Lower quantile for capping.
        upper_quantile (float): Upper quantile for capping.
        new_column (str, optional): Name for the capped column.

    Returns:
        pd.DataFrame: DataFrame with new capped column.
    """
    df = df.copy()
    lower = df[column].quantile(lower_quantile)
    upper = df[column].quantile(upper_quantile)
    col_out = new_column or f"{column}_capped"
    df[col_out] = df[column].clip(lower=lower, upper=upper)
    return df


def create_outlier_flag(
    df: pd.DataFrame,
    column: str,
    threshold_quantile=0.95,
    flag_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Creates a binary flag indicating whether a value is an outlier
    above a quantile threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): Column to evaluate.
        threshold_quantile (float): Quantile above which values are flagged.
        flag_column (str, optional): Name for the flag column.

    Returns:
        pd.DataFrame: DataFrame with a new binary flag column.
    """
    df = df.copy()
    threshold = df[column].quantile(threshold_quantile)
    col_out = flag_column or f"is_outlier_{column}"
    df[col_out] = (df[column] > threshold).astype(int)
    return df
