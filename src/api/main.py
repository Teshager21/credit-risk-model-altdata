# src/api/main.py

from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"

# Load model from MLflow Model Registry
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])

    # Convert object column to category dtype
    df["ProductCategory"] = df["ProductCategory"].astype("category")

    prediction = model.predict(df)
    prediction_value = (
        float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)
    )
    return PredictionResponse(risk_probability=prediction_value)
