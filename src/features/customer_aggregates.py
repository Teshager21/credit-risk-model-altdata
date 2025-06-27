# src/features/customer_aggregates.py

# import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CustomerAggregates(BaseEstimator, TransformerMixin):
    """
    Computes aggregate features per CustomerId and merges them back
    to the original DataFrame.
    """

    def __init__(self, groupby_col="CustomerId"):
        self.groupby_col = groupby_col
        self.agg_df_ = None

    def fit(self, X, y=None):
        agg_df = (
            X.groupby(self.groupby_col)["Amount"]
            .agg(
                total_transaction_amount="sum",
                avg_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std",
            )
            .reset_index()
        )

        # Handle NaN std
        agg_df["std_transaction_amount"] = agg_df["std_transaction_amount"].fillna(0)

        self.agg_df_ = agg_df
        return self

    def transform(self, X):
        X_merged = X.merge(self.agg_df_, on=self.groupby_col, how="left")
        return X_merged
