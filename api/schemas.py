from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        min_length=60,
        description="Lista com pelo menos 60 preços históricos de fechamento (ordem cronológica).",
        examples=[[100.5, 101.2, 99.8]],
    )
    days_ahead: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Número de dias futuros a prever (1–30).",
    )


class PredictResponse(BaseModel):
    ticker: str
    predictions: List[float]
    days_ahead: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ticker: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_requests: int
    avg_response_time_ms: float
    cpu_percent: float
    memory_mb: float
