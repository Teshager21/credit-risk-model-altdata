# src/api/pydantic_models.py
import datetime
from pydantic import BaseModel


class CustomerData(BaseModel):
    TransactionStartTime: datetime.datetime
    Amount: float
    ChannelId: str
    ProductId: str
    ProductCategory: str
    PricingStrategy: int


class PredictionResponse(BaseModel):
    risk_probability: float
