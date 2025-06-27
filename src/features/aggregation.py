# src/transformers/aggregation.py

from sklearn.base import BaseEstimator, TransformerMixin

# import pandas as pd


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col="CustomerId", agg_col="Amount"):
        self.groupby_col = groupby_col
        self.agg_col = agg_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        grouped = (
            X.groupby(self.groupby_col)[self.agg_col]
            .agg(
                Total_Transaction_Amount="sum",
                Avg_Transaction_Amount="mean",
                Transaction_Count="count",
                Std_Transaction_Amount="std",
            )
            .fillna(0)
            .reset_index()
        )
        return grouped
