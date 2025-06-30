# src/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow

# import pickle

# Import your feature pipeline if loaded separately
# from src.features.feature_engineering import build_feature_pipeline

# Load your pipeline and model
import joblib
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
pipeline_path = os.path.join(project_root, "src", "models", "feature_pipeline.joblib")

feature_pipeline = joblib.load(pipeline_path)


# Example loading MLflow model:
# ✅ Replace this with your real run ID
run_id = "ffa7ac57d11147d4a8707c15bbebf337"
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)


# ✅ FULL Request Model with All Required Columns
class PredictionRequest(BaseModel):
    CustomerId: str
    TransactionStartTime: str
    Amount: float
    ChannelId: str
    ProductId: str
    ProductCategory: str
    PricingStrategy: int
    ProviderId: str
    Value: float
    Amount_log: float
    Amount_capped: float
    is_large_transaction: int


app = FastAPI()


@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert request to DataFrame
    input_df = pd.DataFrame([request.dict()])

    # Transform with pipeline
    X_transformed = feature_pipeline.transform(input_df)

    # Predict with model
    prediction = model.predict(X_transformed)

    return {"prediction": int(prediction[0])}
