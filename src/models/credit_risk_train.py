# src/models/credit_risk_train.py

import os
import joblib
import pandas as pd

from features.preprocess import drop_leakage_cols, apply_smote
from data.load_data import (
    load_processed_data,
)  # Should load raw-ish cleaned data with CustomerId etc.
from features.feature_engineering import build_feature_pipeline

from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.gradient_boosting import train_gradient_boosting
from models.xgboost_model import train_xgboost

from utils.metrics import evaluate_model, threshold_tuning, print_classification_report
from utils.mlflow_utils import setup_mlflow, log_model_mlflow

from sklearn.model_selection import train_test_split


# -----------------------------------------------
# Paths and Load Raw-ish Cleaned Data (for feature engineering)
# -----------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(project_root, "data", "interim", "clean_data.parquet")

X, y = load_processed_data(
    data_path
)  # Load data including 'CustomerId', 'Amount', 'TransactionStartTime' etc.

# -----------------------------------------------
# Split Data
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------
# Drop Leakage Columns ONLY
# -----------------------------------------------
cols_to_drop = ["FraudResult"]
X_train_clean = drop_leakage_cols(X_train, cols_to_drop)
X_test_clean = drop_leakage_cols(X_test, cols_to_drop)

print("Columns in X_train_clean before pipeline fit:", X_train_clean.columns.tolist())

# -----------------------------------------------
# Build and Fit Feature Pipeline
# -----------------------------------------------
feature_pipeline = build_feature_pipeline()

feature_pipeline.fit(X_train_clean, y_train)

# Save the pipeline
pipeline_path = os.path.join(project_root, "src", "models", "feature_pipeline.joblib")
joblib.dump(feature_pipeline, pipeline_path)
print(f"Feature pipeline saved to {pipeline_path}")

# -----------------------------------------------
# Transform Data Using Pipeline
# -----------------------------------------------
X_train_transformed = feature_pipeline.transform(X_train_clean)
X_test_transformed = feature_pipeline.transform(X_test_clean)

# Optional: convert transformed arrays to DataFrames with feature names if needed
# Uncomment if your pipeline supports get_feature_names_out
# (some custom transformers might not)
# feature_names = feature_pipeline.named_steps["preprocessor"].get_feature_names_out()
# X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
# X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# -----------------------------------------------
# Save Processed Data for Future Use (Optional)
# -----------------------------------------------
processed_data_path = os.path.join(
    project_root, "data", "processed", "processed_data.parquet"
)
pd.DataFrame(X_train_transformed).to_parquet(processed_data_path)
print(f"Processed training data saved to {processed_data_path}")

# -----------------------------------------------
# Apply SMOTE to Training Data
# -----------------------------------------------
X_train_smote, y_train_smote = apply_smote(X_train_transformed, y_train)

# -----------------------------------------------
# Train and Evaluate Models
# -----------------------------------------------

# Logistic Regression (Optional)
lr_model = train_logistic_regression(X_train_smote, y_train_smote)
y_prob_lr = lr_model.predict_proba(X_test_transformed)[:, 1]
best_thresh_lr, best_f1_lr = threshold_tuning(y_test, y_prob_lr)
y_pred_lr = (y_prob_lr >= best_thresh_lr).astype(int)
lr_metrics = evaluate_model(y_test, y_pred_lr, y_prob_lr)
print("Logistic Regression Metrics:", lr_metrics)
print_classification_report(y_test, y_pred_lr)

# Random Forest
param_grid_rf = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}
best_rf, best_params_rf = train_random_forest(
    X_train_smote, y_train_smote, param_grid_rf
)
y_prob_rf = best_rf.predict_proba(X_test_transformed)[:, 1]
best_thresh_rf, best_f1_rf = threshold_tuning(y_test, y_prob_rf)
y_pred_rf = (y_prob_rf >= best_thresh_rf).astype(int)
rf_metrics = evaluate_model(y_test, y_pred_rf, y_prob_rf)
print("\nRandom Forest Metrics:", rf_metrics)
print_classification_report(y_test, y_pred_rf)

# Gradient Boosting
param_grid_gbm = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8],
}
best_gbm, best_params_gbm = train_gradient_boosting(
    X_train_smote, y_train_smote, param_grid_gbm
)
y_prob_gbm = best_gbm.predict_proba(X_test_transformed)[:, 1]
best_thresh_gbm, best_f1_gbm = threshold_tuning(y_test, y_prob_gbm)
y_pred_gbm = (y_prob_gbm >= best_thresh_gbm).astype(int)
gbm_metrics = evaluate_model(y_test, y_pred_gbm, y_prob_gbm)
print("\nGradient Boosting Metrics:", gbm_metrics)
print_classification_report(y_test, y_pred_gbm)

# XGBoost
param_grid_xgb = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "scale_pos_weight": [1, 2, 5],
}
best_xgb, best_params_xgb = train_xgboost(X_train_smote, y_train_smote, param_grid_xgb)
y_prob_xgb = best_xgb.predict_proba(X_test_transformed)[:, 1]
best_thresh_xgb, best_f1_xgb = threshold_tuning(y_test, y_prob_xgb)
y_pred_xgb = (y_prob_xgb >= best_thresh_xgb).astype(int)
xgb_metrics = evaluate_model(y_test, y_pred_xgb, y_prob_xgb)
print("\nXGBoost Metrics:", xgb_metrics)
print_classification_report(y_test, y_pred_xgb)

# -----------------------------------------------
# MLflow Logging
# -----------------------------------------------
setup_mlflow()

rf_run_id = log_model_mlflow(
    best_rf,
    model_name="CreditRiskModel_RF",
    params=best_params_rf,
    metrics=rf_metrics,
    threshold=best_thresh_rf,
    run_name="RandomForest_SMOTE_ThresholdTuned",
    flavor="sklearn",
    register=True,
    stage="Production",
)

xgb_run_id = log_model_mlflow(
    best_xgb,
    model_name="CreditRiskModel_XGB",
    params=best_params_xgb,
    metrics=xgb_metrics,
    threshold=best_thresh_xgb,
    run_name="XGBoost_SMOTE_ThresholdTuned",
    flavor="xgboost",
    register=True,
    stage="Production",
)

print(f"Random Forest run ID: {rf_run_id}")
print(f"XGBoost run ID: {xgb_run_id}")
