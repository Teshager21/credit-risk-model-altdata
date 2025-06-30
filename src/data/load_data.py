import pandas as pd


def load_processed_data(filepath):
    df = pd.read_parquet(filepath)
    target_col = "FraudResult"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    y = df[target_col]
    X = df.drop(columns=[target_col], errors="ignore")
    return X, y
