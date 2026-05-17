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


class ModelMetricsResponse(BaseModel):
    ticker: str = Field(..., description="Ativo financeiro utilizado no treinamento.")
    model_version: str = Field(..., description="Versão do modelo treinado.")
    trained_at: str = Field(..., description="Data e hora em que o modelo foi treinado.")
    train_period: str = Field(..., description="Período histórico utilizado no treinamento.")
    architecture: str = Field(..., description="Arquitetura das camadas da rede neural.")
    optimizer: str = Field(..., description="Algoritmo de otimização utilizado.")
    loss_function: str = Field(..., description="Função de perda utilizada no treinamento.")
    total_records: int = Field(..., description="Total de pregões baixados do Yahoo Finance.")
    train_samples: int = Field(..., description="Quantidade de amostras usadas no treino.")
    test_samples: int = Field(..., description="Quantidade de amostras usadas na avaliação.")
    sequence_length: int = Field(..., description="Janela de dias usada como entrada do modelo.")
    epochs_executed: int = Field(..., description="Número de épocas até o early stopping.")
    mae: float = Field(..., description="Mean Absolute Error na escala normalizada (0–1).")
    rmse: float = Field(..., description="Root Mean Squared Error na escala normalizada (0–1).")
    mape: float = Field(..., description="Mean Absolute Percentage Error em % (quanto menor, melhor).")
    mape_rating: str = Field(..., description="Avaliação qualitativa da performance do modelo.")
