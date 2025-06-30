# src/api/pydantic_models.py

from pydantic import BaseModel


class CustomerData(BaseModel):
    # Replace with your real model's features!
    feature1: float
    feature2: float
    feature3: float


class PredictionResponse(BaseModel):
    risk_probability: float
