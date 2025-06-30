import pandas as pd


def load_processed_data(path):
    df = pd.read_parquet(path)
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]
    return X, y
