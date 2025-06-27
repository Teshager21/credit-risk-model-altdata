# src/features/manual_woe_encoder.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ManualWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features, target="FraudResult"):
        self.features = features
        self.target = target
        self.woe_maps = {}

    def fit(self, X, y):
        Xy = X.copy()
        Xy[self.target] = y

        for feature in self.features:
            df = (
                Xy.groupby(feature)[self.target]
                .agg(["sum", "count"])
                .rename(columns={"sum": "bad"})
            )
            df["good"] = df["count"] - df["bad"]
            total_bad = df["bad"].sum()
            total_good = df["good"].sum()

            df["woe"] = np.log(
                ((df["good"] / total_good) + 1e-6) / ((df["bad"] / total_bad) + 1e-6)
            )
            self.woe_maps[feature] = df["woe"].to_dict()

        return self

    def transform(self, X):
        X_ = X.copy()
        for feature in self.features:
            X_[f"{feature}_woe"] = X_[feature].map(self.woe_maps[feature]).fillna(0)
        return X_.drop(columns=self.features)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_woe" for col in self.features])
