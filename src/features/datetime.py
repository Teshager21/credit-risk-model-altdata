# src/transformers/datetime.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class DateTimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.column] = pd.to_datetime(X_[self.column])
        X_["transaction_hour"] = X_[self.column].dt.hour
        X_["transaction_day"] = X_[self.column].dt.day
        X_["transaction_month"] = X_[self.column].dt.month
        X_["transaction_weekday"] = X_[self.column].dt.weekday
        X_["transaction_year"] = X_[self.column].dt.year
        return X_[
            [
                "transaction_hour",
                "transaction_day",
                "transaction_month",
                "transaction_weekday",
                "transaction_year",
            ]
        ]

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                f"{self.column}_hour",
                f"{self.column}_day",
                f"{self.column}_month",
                f"{self.column}_weekday",
                f"{self.column}_year",
            ]
        )
