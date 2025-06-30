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
    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)

    # Ensure it's a float, even if model returns an array
    if hasattr(prediction, "__len__"):
        prediction_value = float(prediction[0])
    else:
        prediction_value = float(prediction)

    return PredictionResponse(risk_probability=prediction_value)
