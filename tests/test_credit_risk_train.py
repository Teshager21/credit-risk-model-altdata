import os
import tempfile
import numpy as np
import pandas as pd
import joblib

from src.features.feature_engineering import build_feature_pipeline
from src.features.preprocess import apply_smote, drop_leakage_cols
from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest
from src.models.gradient_boosting import train_gradient_boosting
from src.models.xgboost_model import train_xgboost
from src.utils.metrics import evaluate_model, threshold_tuning

from sklearn.model_selection import train_test_split


def test_credit_risk_training_pipeline():
    # -----------------------------
    # Create mock data
    # -----------------------------
    n_samples = 20

    # Minimal example data matching your feature pipeline
    df = pd.DataFrame(
        {
            "CustomerId": [f"cust_{i}" for i in range(n_samples)],
            "TransactionStartTime": pd.date_range(
                start="2023-01-01", periods=n_samples, freq="H"
            ),
            "Amount": np.random.rand(n_samples) * 100,
            "Value": np.random.rand(n_samples) * 50,
            "Amount_log": np.log1p(np.random.rand(n_samples) * 100),
            "Amount_capped": np.random.rand(n_samples) * 100,
            "ProductCategory": np.random.choice(["A", "B"], n_samples),
            "ChannelId": np.random.choice(["X", "Y"], n_samples),
            "ProviderId": np.random.choice(["P1", "P2"], n_samples),
            "ProductId": np.random.choice(["PR1", "PR2"], n_samples),
            "PricingStrategy": np.random.choice([1, 2], n_samples),
            "is_large_transaction": np.random.choice([0, 1], n_samples),
        }
    )

    # Binary target
    y = np.random.choice([0, 1], n_samples)

    # -----------------------------
    # Split data
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Drop leakage columns
    # -----------------------------
    cols_to_drop = ["FraudResult"]
    X_train_clean = drop_leakage_cols(X_train, cols_to_drop)
    X_test_clean = drop_leakage_cols(X_test, cols_to_drop)

    # -----------------------------
    # Build feature pipeline
    # -----------------------------
    pipeline = build_feature_pipeline()
    pipeline.fit(X_train_clean, y_train)

    # Test saving pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "feature_pipeline.joblib")
        joblib.dump(pipeline, path)
        assert os.path.exists(path)

    # Transform data
    X_train_transformed = pipeline.transform(X_train_clean)
    X_test_transformed = pipeline.transform(X_test_clean)

    assert X_train_transformed.shape[0] == X_train_clean.shape[0]
    assert not np.isnan(X_train_transformed).any()

    # -----------------------------
    # SMOTE
    # -----------------------------
    X_train_smote, y_train_smote = apply_smote(X_train_transformed, y_train)
    assert X_train_smote.shape[0] >= X_train_transformed.shape[0]

    # -----------------------------
    # Train logistic regression
    # -----------------------------
    model_lr = train_logistic_regression(X_train_smote, y_train_smote)
    y_prob_lr = model_lr.predict_proba(X_test_transformed)[:, 1]
    thresh_lr, f1_lr = threshold_tuning(y_test, y_prob_lr)
    y_pred_lr = (y_prob_lr >= thresh_lr).astype(int)
    metrics_lr = evaluate_model(y_test, y_pred_lr, y_prob_lr)
    assert "f1" in metrics_lr

    # -----------------------------
    # Train random forest
    # -----------------------------
    param_grid_rf = {
        "n_estimators": [10],
        "max_depth": [3],
        "min_samples_split": [2],
    }
    model_rf, params_rf = train_random_forest(
        X_train_smote, y_train_smote, param_grid_rf
    )
    y_prob_rf = model_rf.predict_proba(X_test_transformed)[:, 1]
    assert y_prob_rf.shape == (X_test_transformed.shape[0],)

    # -----------------------------
    # Train gradient boosting
    # -----------------------------
    param_grid_gbm = {
        "n_estimators": [10],
        "learning_rate": [0.1],
        "max_depth": [3],
        "subsample": [0.8],
    }
    model_gbm, params_gbm = train_gradient_boosting(
        X_train_smote, y_train_smote, param_grid_gbm
    )
    y_prob_gbm = model_gbm.predict_proba(X_test_transformed)[:, 1]
    assert y_prob_gbm.shape == (X_test_transformed.shape[0],)

    # -----------------------------
    # Train XGBoost
    # -----------------------------
    param_grid_xgb = {
        "n_estimators": [10],
        "max_depth": [3],
        "learning_rate": [0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "scale_pos_weight": [1],
    }
    model_xgb, params_xgb = train_xgboost(X_train_smote, y_train_smote, param_grid_xgb)
    y_prob_xgb = model_xgb.predict_proba(X_test_transformed)[:, 1]
    assert y_prob_xgb.shape == (X_test_transformed.shape[0],)

    print("✅ credit_risk_train pipeline test completed successfully.")
