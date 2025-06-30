# src/features/customer_aggregates.py

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# import pandas as pd


class CustomerAggregates(BaseEstimator, TransformerMixin):
    """
    Computes aggregate features per CustomerId and merges them back
    to the original DataFrame.
    """

    def __init__(self, groupby_col="CustomerId"):
        self.groupby_col = groupby_col
        self.agg_df_ = None

    def fit(self, X, y=None):
        # Check if groupby_col exists in X
        if self.groupby_col not in X.columns:
            raise KeyError(
                f"Column '{self.groupby_col}' not found in input data during fit."
            )

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
        if self.agg_df_ is None:
            raise RuntimeError(
                "CustomerAggregates transformer has not been fitted yet. "
                "Call .fit or .fit_transform first."
            )

        if self.groupby_col not in X.columns:
            # Instead of crashing, add zeros for aggregate columns
            print(
                f"[WARN] Column '{self.groupby_col}' not found in input data."
                "Filling aggregates with default zeros."
            )
            X["total_transaction_amount"] = 0
            X["avg_transaction_amount"] = 0
            X["transaction_count"] = 0
            X["std_transaction_amount"] = 0
            return X

        # Merge aggregate features
        X_merged = X.merge(self.agg_df_, on=self.groupby_col, how="left")

        # Fill any missing aggregate values with zeros
        X_merged["total_transaction_amount"] = X_merged[
            "total_transaction_amount"
        ].fillna(0)
        X_merged["avg_transaction_amount"] = X_merged["avg_transaction_amount"].fillna(
            0
        )

        X_merged["transaction_count"] = X_merged["transaction_count"].fillna(0)
        X_merged["std_transaction_amount"] = X_merged["std_transaction_amount"].fillna(
            0
        )

        return X_merged

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "total_transaction_amount",
                "avg_transaction_amount",
                "std_transaction_amount",
                "transaction_count",
            ]
        )
