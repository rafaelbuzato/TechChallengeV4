import os
import time
import numpy as np
import joblib
from contextlib import asynccontextmanager

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import PredictRequest, PredictResponse, HealthResponse, MetricsResponse
from monitoring.middleware import MonitoringMiddleware, get_metrics

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "lstm_model.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "scaler.pkl")
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

    return PredictResponse(
        ticker=TICKER,
        predictions=[round(p, 4) for p in pred_inv],
        days_ahead=request.days_ahead,
        model_version=MODEL_VERSION,
    )
