import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from path"""
    df = pd.read_csv(path)
    print(f"✅ Loaded data with shape: {df.shape}")
    return df
