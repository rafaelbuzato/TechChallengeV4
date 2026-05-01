import os
import sys

# Força UTF-8 no stdout para evitar erros de encoding no Windows (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Configurações ────────────────────────────────────────────────────────────
TICKER = "PETR4.SA"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
SEQUENCE_LENGTH = 60       # janela de dias usada como entrada
TRAIN_RATIO = 0.8
EPOCHS = 100
BATCH_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lstm_model.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "petr4_raw.csv")


def download_data() -> pd.DataFrame:
    print(f"Baixando dados de {TICKER} ({START_DATE} → {END_DATE})...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.to_csv(DATA_PATH)
    print(f"Dados salvos em {DATA_PATH} — {len(df)} registros.")
    return df


def build_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_model(seq_len: int) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n── {label} Metrics ──────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def plot_results(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    plt.figure(figsize=(14, 5))
    plt.plot(y_true_inv, label="Real", linewidth=1.5)
    plt.plot(y_pred_inv, label="Previsto", linewidth=1.5, linestyle="--")
    plt.title(f"PETR4.SA — Previsão de Fechamento (LSTM)")
    plt.xlabel("Dias")
    plt.ylabel("Preço (R$)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "data", "prediction_plot.png")
    plt.savefig(out, dpi=150)
    print(f"\nGráfico salvo em {out}")


def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    df = download_data()
    close = df[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvo em {SCALER_PATH}")

    split = int(len(scaled) * TRAIN_RATIO)
    train_data = scaled[:split]
    test_data = scaled[split - SEQUENCE_LENGTH:]

    X_train, y_train = build_sequences(train_data, SEQUENCE_LENGTH)
    X_test, y_test = build_sequences(test_data, SEQUENCE_LENGTH)

    X_train = X_train.reshape(-1, SEQUENCE_LENGTH, 1)
    X_test = X_test.reshape(-1, SEQUENCE_LENGTH, 1)

    print(f"\nTreino: {X_train.shape} | Teste: {X_test.shape}")

    model = build_model(SEQUENCE_LENGTH)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss"),
    ]

    print("\nIniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred.flatten())

    plot_results(y_test, y_pred.flatten(), scaler)

    print(f"\nModelo salvo em {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    main()
