from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    prices: List[float] = Field(
        ...,
        min_length=60,
        description="Lista com pelo menos 60 preços históricos de fechamento (ordem cronológica).",
        examples=[[
            30.77, 31.14, 31.06, 31.5,  30.87, 30.55, 30.91, 30.88, 30.96, 30.7,
            30.55, 30.32, 30.23, 29.76, 29.64, 29.27, 29.47, 29.68, 29.63, 29.56,
            29.43, 29.48, 29.08, 29.14, 29.05, 29.06, 29.15, 29.7,  29.76, 30.32,
            30.27, 30.59, 31.36, 31.03, 31.12, 32.36, 32.16, 32.12, 32.01, 31.68,
            31.93, 32.14, 32.43, 32.22, 32.54, 32.04, 32.87, 32.99, 33.32, 32.72,
            32.52, 32.38, 32.69, 31.85, 31.72, 31.45, 31.46, 31.7,  31.6,  32.07,
        ]],
    )
    days_ahead: int = Field(
        default=5,
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
