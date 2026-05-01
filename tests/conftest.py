"""
conftest.py — fixtures e mocks globais para todo o suite de testes.

Estratégia de mock:
  - tensorflow/keras são simulados via sys.modules antes de qualquer import,
    pois podem não estar instalados (Python 3.13 ainda sem suporte oficial).
  - O modelo LSTM é substituído por um MagicMock com predict() configurado.
  - O scaler MinMaxScaler é criado de verdade com dados sintéticos.
"""

import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import MinMaxScaler
from fastapi.testclient import TestClient


# ── Mock completo de tensorflow / keras ──────────────────────────────────────

def _make_tf_mock():
    tf_mock = MagicMock(name="tensorflow")
    keras_mock = MagicMock(name="keras")

    # keras.models.load_model
    keras_mock.models = MagicMock()
    keras_mock.models.Sequential = MagicMock()

    # keras.layers
    keras_mock.layers = MagicMock()

    # keras.callbacks
    keras_mock.callbacks = MagicMock()

    tf_mock.keras = keras_mock
    tf_mock.__version__ = "2.16.1 (mock)"

    return tf_mock


_tf_mock = _make_tf_mock()

# Registrar antes de qualquer import do projeto
for name in ["tensorflow", "tensorflow.keras", "tensorflow.keras.models",
             "tensorflow.keras.layers", "tensorflow.keras.callbacks"]:
    sys.modules.setdefault(name, _tf_mock)


# ── Dados sintéticos PETR4-like ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_prices():
    """Série temporal sintética de 500 dias com padrão realista."""
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    prices = 30 + 10 * np.sin(t) + np.random.normal(0, 1, 500)
    return prices.astype(np.float32)


@pytest.fixture(scope="session")
def fitted_scaler(raw_prices):
    """MinMaxScaler ajustado nos dados sintéticos."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(raw_prices.reshape(-1, 1))
    return scaler


@pytest.fixture(scope="session")
def scaled_prices(raw_prices, fitted_scaler):
    return fitted_scaler.transform(raw_prices.reshape(-1, 1))


@pytest.fixture(scope="session")
def mock_keras_model(fitted_scaler):
    """Modelo Keras simulado que devolve previsão fixa (0.5 no espaço escalado)."""
    model = MagicMock()
    # predict sempre retorna 0.5 (espaço normalizado) → ~R$40 após inverse_transform
    model.predict.return_value = np.array([[0.5]])
    return model


# ── TestClient da API com mocks injetados ────────────────────────────────────

@pytest.fixture(scope="module")
def api_client(mock_keras_model, fitted_scaler):
    """
    TestClient do FastAPI com model e scaler injetados diretamente,
    sem precisar carregar arquivos do disco.

    IMPORTANTE: a injeção é feita DEPOIS de entrar no contexto do TestClient,
    pois a lifespan do FastAPI roda ao entrar e pode sobrescrever as variáveis
    globais de model/scaler caso tente carregar os arquivos do disco.
    """
    import api.main as main_module

    with TestClient(main_module.app) as client:
        # Injeta APÓS a lifespan ter rodado (e falhado silenciosamente no load)
        main_module.model = mock_keras_model
        main_module.scaler = fitted_scaler
        yield client
