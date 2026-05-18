import os
import json
import time
import numpy as np
import joblib
from contextlib import asynccontextmanager

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import PredictRequest, PredictResponse, DayPrediction, HealthResponse, MetricsResponse, ModelMetricsResponse
from monitoring.middleware import MonitoringMiddleware, get_metrics

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "model", "lstm_model.keras")
SCALER_PATH  = os.path.join(os.path.dirname(__file__), "..", "model", "scaler.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")
TICKER = "PETR4.SA"
MODEL_VERSION = "1.0.0"
SEQUENCE_LENGTH = 60

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Modelo e scaler carregados com sucesso.")
    except Exception as e:
        print(f"AVISO: modelo não carregado — {e}")
    yield


app = FastAPI(
    title="PETR4 LSTM Predictor",
    description="API para previsão do preço de fechamento da PETR4.SA usando LSTM.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MonitoringMiddleware)

Instrumentator().instrument(app).expose(app)


@app.get("/", tags=["Root"])
def root():
    return {"message": "PETR4 LSTM Predictor — acesse /docs para a documentação."}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        ticker=TICKER,
    )


@app.get("/metrics-summary", response_model=MetricsResponse, tags=["Monitoring"])
def metrics_summary():
    return get_metrics()


@app.get("/model-metrics", response_model=ModelMetricsResponse, tags=["Model"])
def model_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Arquivo de métricas não encontrado. Execute o treinamento primeiro.",
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelMetricsResponse(**data)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo não está carregado.")

    if len(request.prices) < SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"São necessários pelo menos {SEQUENCE_LENGTH} preços históricos.",
        )

    prices = np.array(request.prices[-SEQUENCE_LENGTH:]).reshape(-1, 1)
    scaled = scaler.transform(prices)

    predictions = []
    window = scaled.copy()

    for _ in range(request.days_ahead):
        x = window[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 1)
        pred_scaled = model.predict(x, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        window = np.append(window, [[pred_scaled]], axis=0)

    pred_array = np.array(predictions).reshape(-1, 1)
    pred_inv = scaler.inverse_transform(pred_array).flatten().tolist()

    reference_price = round(request.prices[-1], 4)

    day_predictions = []
    for i, price in enumerate(pred_inv):
        price = round(price, 4)
        prev = reference_price if i == 0 else round(pred_inv[i - 1], 4)
        change_pct = round((price - prev) / prev * 100, 2)
        direction = "Alta" if change_pct > 0.05 else ("Baixa" if change_pct < -0.05 else "Estável")
        day_predictions.append(DayPrediction(
            day=i + 1,
            label=f"Dia {i + 1}",
            price=price,
            change_pct=change_pct,
            direction=direction,
        ))

    first_price = round(pred_inv[0], 4)
    last_price  = round(pred_inv[-1], 4)
    overall_change = (last_price - reference_price) / reference_price * 100
    trend = "Alta" if overall_change > 0.1 else ("Baixa" if overall_change < -0.1 else "Lateral")

    return PredictResponse(
        ticker=TICKER,
        model_version=MODEL_VERSION,
        days_ahead=request.days_ahead,
        reference_price=reference_price,
        trend=trend,
        predictions=day_predictions,
    )
