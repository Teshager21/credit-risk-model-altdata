from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from features.datetime import DateTimeFeaturesExtractor
from features.encoder import ManualWOEEncoder
from features.customer_aggregates import CustomerAggregates


def build_feature_pipeline():
    datetime_col = "TransactionStartTime"

    numeric_features = [
        "Amount",
        "Value",
        "Amount_log",
        "Amount_capped",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    woe_features = ["ProductCategory", "ChannelId", "ProviderId", "ProductId"]

    onehot_features = ["PricingStrategy", "is_large_transaction"]

    onehot_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "datetime",
                DateTimeFeaturesExtractor(column=datetime_col),
                [datetime_col],
            ),
            ("numeric", numeric_transformer, numeric_features),
            ("woe", ManualWOEEncoder(features=woe_features), woe_features),
            ("onehot", onehot_transformer, onehot_features),
        ]
    )

    pipeline = Pipeline(
        [
            ("customer_agg", CustomerAggregates(groupby_col="CustomerId")),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline
