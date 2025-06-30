# src/api/pydantic_models.py

from pydantic import BaseModel


class CustomerData(BaseModel):
    Amount: float
    ProductCategory: str


class PredictionResponse(BaseModel):
    risk_probability: float
