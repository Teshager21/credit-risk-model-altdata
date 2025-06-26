"""
summary_stats.py

Module to extract basic summary statistics and structure information
from a Pandas DataFrame for exploratory data analysis (EDA).
"""

import pandas as pd
from typing import Dict, Any
from io import StringIO


def get_dataframe_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return key summary information about the structure and content of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - shape: Tuple of (rows, columns)
            - columns: List of column names
            - dtypes: Dictionary of column data types
            - info: String representation of df.info()
            - head: First few rows (as DataFrame)
            - describe: Summary stats (as DataFrame)
    """
    # Capture df.info() output
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    buffer.close()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "info": info_str,
        "head": df.head(3),
        "describe": df.describe(include="all"),
    }
