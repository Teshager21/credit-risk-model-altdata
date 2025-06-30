import pandas as pd
import numpy as np

from src.features.feature_engineering import build_feature_pipeline


def test_build_feature_pipeline_fit_transform():
    # Test input WITHOUT aggregate columns
    data = {
        "CustomerId": ["cust1", "cust2"],
        "TransactionStartTime": ["2023-01-01 12:00:00", "2023-01-02 13:30:00"],
        "Amount": [100.0, 150.0],
        "Value": [50.0, 75.0],
        "Amount_log": [4.6, 5.0],
        "Amount_capped": [100.0, 150.0],
        "ProductCategory": ["A", "B"],
        "ChannelId": ["X", "Y"],
        "ProviderId": ["P1", "P2"],
        "ProductId": ["PR1", "PR2"],
        "PricingStrategy": [1, 2],
        "is_large_transaction": [0, 1],
    }

    df = pd.DataFrame(data)

    # Add a fake target
    y = pd.Series([0, 1])

    pipeline = build_feature_pipeline()

    # ✅ Provide y
    pipeline.fit(df, y)

    transformed = pipeline.transform(df)

    # Check that transformation works
    assert transformed.shape[0] == df.shape[0]
    assert not np.any(pd.isna(transformed))
