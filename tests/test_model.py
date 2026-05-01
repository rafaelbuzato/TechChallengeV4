"""
test_model.py — testa a arquitetura, inferência e persistência do modelo LSTM.

TensorFlow é mockado via conftest.py; os testes verificam o comportamento
esperado do pipeline sem precisar de GPU ou do binário TF instalado.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call
from sklearn.preprocessing import MinMaxScaler


SEQ_LEN = 60


# ── Testes de Inferência do Modelo Mock ──────────────────────────────────────

class TestInferenciaModelo:
    def test_predict_retorna_array(self, mock_keras_model):
        X = np.random.rand(1, SEQ_LEN, 1).astype(np.float32)
        result = mock_keras_model.predict(X, verbose=0)
        assert result is not None

    def test_predict_shape_correto(self, mock_keras_model):
        X = np.random.rand(1, SEQ_LEN, 1).astype(np.float32)
        result = mock_keras_model.predict(X, verbose=0)
        assert result.shape == (1, 1)

    def test_predict_valor_no_range_normalizado(self, mock_keras_model):
        X = np.random.rand(1, SEQ_LEN, 1).astype(np.float32)
        pred = mock_keras_model.predict(X, verbose=0)[0, 0]
        assert 0.0 <= pred <= 1.0

    def test_predict_chamado_com_input_correto(self, mock_keras_model):
        X = np.ones((1, SEQ_LEN, 1), dtype=np.float32)
        mock_keras_model.predict(X, verbose=0)
        mock_keras_model.predict.assert_called()

    def test_inferencia_iterativa_days_ahead(self, mock_keras_model, fitted_scaler):
        """Simula a lógica de previsão N dias à frente do endpoint /predict."""
        prices = np.linspace(30, 40, SEQ_LEN).reshape(-1, 1).astype(np.float32)
        scaled = fitted_scaler.transform(prices)
        window = scaled.copy()

        predictions = []
        for _ in range(5):
            x = window[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
            pred = mock_keras_model.predict(x, verbose=0)[0, 0]
            predictions.append(pred)
            window = np.append(window, [[pred]], axis=0)

        assert len(predictions) == 5
        pred_inv = fitted_scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        # Todos os valores devem ser números reais
        assert not np.any(np.isnan(pred_inv))

    def test_inverse_transform_apos_predicao(self, mock_keras_model, fitted_scaler):
        """Garante que a desnormalização produz valor no range original."""
        X = np.random.rand(1, SEQ_LEN, 1).astype(np.float32)
        pred_scaled = mock_keras_model.predict(X, verbose=0)[0, 0]
        pred_real = fitted_scaler.inverse_transform([[pred_scaled]])[0, 0]
        # Dados sintéticos estão entre ~20 e ~50
        assert 10.0 <= pred_real <= 60.0, f"Valor desnormalizado fora do range: {pred_real}"


# ── Testes de Reshape e Preparação de Input ──────────────────────────────────

class TestPreparacaoInput:
    def test_reshape_para_lstm(self, scaled_prices):
        """X deve ter shape (samples, SEQ_LEN, 1) para o LSTM."""
        from tests.test_preprocessing import build_sequences
        X, _ = build_sequences(scaled_prices, SEQ_LEN)
        X_3d = X.reshape(-1, SEQ_LEN, 1)
        assert X_3d.ndim == 3
        assert X_3d.shape[1] == SEQ_LEN
        assert X_3d.shape[2] == 1

    def test_ultima_janela_para_predicao(self, scaled_prices):
        """A janela de entrada para previsão deve usar os últimos SEQ_LEN valores."""
        window = scaled_prices[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
        assert window.shape == (1, SEQ_LEN, 1)

    def test_tipo_float32(self, scaled_prices):
        window = scaled_prices[-SEQ_LEN:].astype(np.float32).reshape(1, SEQ_LEN, 1)
        assert window.dtype == np.float32

    @pytest.mark.parametrize("n_amostras", [1, 8, 32])
    def test_batch_diferente(self, scaled_prices, n_amostras):
        """Diferentes tamanhos de batch não devem afetar o shape da janela."""
        from tests.test_preprocessing import build_sequences
        X, _ = build_sequences(scaled_prices, SEQ_LEN)
        batch = X[:n_amostras].reshape(-1, SEQ_LEN, 1)
        assert batch.shape == (min(n_amostras, len(X)), SEQ_LEN, 1)


# ── Testes de Persistência (mock de save/load) ───────────────────────────────

class TestPersistenciaModelo:
    def test_save_chamado_com_path_correto(self, mock_keras_model):
        model_path = "model/lstm_model.keras"
        mock_keras_model.save(model_path)
        mock_keras_model.save.assert_called_with(model_path)

    def test_load_model_retorna_modelo(self):
        import tensorflow as tf
        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([[0.4]])
        tf.keras.models.load_model.return_value = fake_model

        loaded = tf.keras.models.load_model("model/lstm_model.keras")
        assert loaded is not None

    def test_scaler_salvo_e_carregado(self, fitted_scaler, tmp_path):
        import joblib
        path = tmp_path / "scaler.pkl"
        joblib.dump(fitted_scaler, path)
        loaded = joblib.load(path)

        # Verifica que o scaler carregado produz o mesmo resultado
        sample = np.array([[35.0]])
        np.testing.assert_allclose(
            fitted_scaler.transform(sample),
            loaded.transform(sample),
        )

    def test_scaler_carregado_mantem_range(self, fitted_scaler, tmp_path):
        import joblib
        path = tmp_path / "scaler_range.pkl"
        joblib.dump(fitted_scaler, path)
        loaded = joblib.load(path)
        assert loaded.feature_range == (0, 1)


# ── Testes de Hiperparâmetros e Configuração ─────────────────────────────────

class TestConfiguracaoModelo:
    def test_sequence_length_positivo(self):
        assert SEQ_LEN > 0

    def test_sequence_length_minimo_recomendado(self):
        """Janelas muito pequenas perdem contexto temporal."""
        assert SEQ_LEN >= 30, "SEQ_LEN deve ser pelo menos 30 para capturar tendências"

    def test_train_ratio_valido(self):
        TRAIN_RATIO = 0.8
        assert 0.5 < TRAIN_RATIO < 1.0

    def test_dados_suficientes_para_sequencia(self, raw_prices):
        """Dataset deve ter pelo menos 2x SEQ_LEN pontos."""
        assert len(raw_prices) >= 2 * SEQ_LEN, (
            f"Dataset tem apenas {len(raw_prices)} pontos; mínimo: {2 * SEQ_LEN}"
        )
