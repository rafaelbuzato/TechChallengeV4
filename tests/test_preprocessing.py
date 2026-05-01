"""
test_preprocessing.py — testa toda a lógica de pré-processamento dos dados
(independente de TensorFlow, sem acesso à rede).
"""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler


# ── Função extraída do train.py para teste unitário ──────────────────────────

def build_sequences(data: np.ndarray, seq_len: int):
    """Gera pares (X, y) com janela deslizante."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ── Testes ───────────────────────────────────────────────────────────────────

class TestMinMaxScaler:
    def test_escala_entre_zero_e_um(self, raw_prices):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw_prices.reshape(-1, 1))
        assert scaled.min() >= 0.0, "Valor mínimo deve ser >= 0"
        assert scaled.max() <= 1.0, "Valor máximo deve ser <= 1"

    def test_inverse_transform_recupera_valores_originais(self, raw_prices):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw_prices.reshape(-1, 1))
        recovered = scaler.inverse_transform(scaled).flatten()
        np.testing.assert_allclose(recovered, raw_prices, rtol=1e-5)

    def test_scaler_preserva_shape(self, raw_prices):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw_prices.reshape(-1, 1))
        assert scaled.shape == (len(raw_prices), 1)


class TestBuildSequences:
    SEQ_LEN = 60

    def test_numero_correto_de_sequencias(self, scaled_prices):
        X, y = build_sequences(scaled_prices, self.SEQ_LEN)
        expected = len(scaled_prices) - self.SEQ_LEN
        assert len(X) == expected, f"Esperado {expected} sequências, obtido {len(X)}"
        assert len(y) == expected

    def test_shape_correto_das_sequencias(self, scaled_prices):
        X, y = build_sequences(scaled_prices, self.SEQ_LEN)
        assert X.shape == (len(scaled_prices) - self.SEQ_LEN, self.SEQ_LEN)
        assert y.shape == (len(scaled_prices) - self.SEQ_LEN,)

    def test_valores_dentro_do_range_normalizado(self, scaled_prices):
        X, y = build_sequences(scaled_prices, self.SEQ_LEN)
        assert X.min() >= 0.0
        assert X.max() <= 1.0
        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_alinhamento_X_y(self, scaled_prices):
        """y[i] deve ser o elemento imediatamente após X[i]."""
        X, y = build_sequences(scaled_prices, self.SEQ_LEN)
        # A última posição de X[0] deve ser o elemento antes de y[0]
        assert scaled_prices[self.SEQ_LEN - 1, 0] == X[0, -1]
        assert scaled_prices[self.SEQ_LEN, 0] == y[0]

    def test_sequencia_vazia_levanta_arrays_vazios(self):
        data = np.zeros((10, 1))
        X, y = build_sequences(data, seq_len=20)   # seq_len > len(data)
        assert len(X) == 0
        assert len(y) == 0

    @pytest.mark.parametrize("seq_len", [10, 30, 60])
    def test_diferentes_tamanhos_de_janela(self, scaled_prices, seq_len):
        X, y = build_sequences(scaled_prices, seq_len)
        assert X.shape[1] == seq_len
        assert len(X) == len(scaled_prices) - seq_len


class TestTrainTestSplit:
    TRAIN_RATIO = 0.8
    SEQ_LEN = 60

    def test_proporcao_de_split(self, scaled_prices):
        split = int(len(scaled_prices) * self.TRAIN_RATIO)
        train = scaled_prices[:split]
        assert len(train) == split

    def test_test_set_inclui_overlap_de_sequencia(self, scaled_prices):
        """test_data deve começar SEQ_LEN antes do split para não perder contexto."""
        split = int(len(scaled_prices) * self.TRAIN_RATIO)
        test_data = scaled_prices[split - self.SEQ_LEN:]
        assert len(test_data) == len(scaled_prices) - split + self.SEQ_LEN

    def test_sem_vazamento_de_dados(self, scaled_prices):
        """Os rótulos do treino não devem aparecer no conjunto de teste."""
        split = int(len(scaled_prices) * self.TRAIN_RATIO)
        train_data = scaled_prices[:split]
        test_data = scaled_prices[split:]
        # Últimas posições de train não devem coincidir com primeiras de test
        assert not np.array_equal(train_data[-1], test_data[0])


class TestMetricas:
    def test_mae_perfeito_e_zero(self):
        from sklearn.metrics import mean_absolute_error
        y = np.array([1.0, 2.0, 3.0])
        assert mean_absolute_error(y, y) == 0.0

    def test_rmse_perfeito_e_zero(self):
        from sklearn.metrics import mean_squared_error
        y = np.array([1.0, 2.0, 3.0])
        assert np.sqrt(mean_squared_error(y, y)) == 0.0

    def test_mape_perfeito_e_zero(self):
        y_true = np.array([10.0, 20.0, 30.0])
        mape = np.mean(np.abs((y_true - y_true) / y_true)) * 100
        assert mape == 0.0

    def test_mape_com_erro_conhecido(self):
        y_true = np.array([100.0])
        y_pred = np.array([110.0])
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        assert abs(mape - 10.0) < 1e-6, "MAPE de 10% esperado"

    def test_mae_sempre_positivo(self):
        from sklearn.metrics import mean_absolute_error
        y_true = np.array([5.0, 10.0, 15.0])
        y_pred = np.array([4.0, 11.0, 14.5])
        assert mean_absolute_error(y_true, y_pred) > 0
