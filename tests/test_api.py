"""
test_api.py — testa todos os endpoints da API FastAPI.

O model e scaler são injetados via conftest.py (sem arquivo no disco).
"""

import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _prices(n: int = 60) -> list[float]:
    """Gera lista de n preços sintéticos crescentes."""
    np.random.seed(0)
    return (30 + np.cumsum(np.random.randn(n) * 0.5)).tolist()


# ── Testes de Saúde e Rota Raiz ──────────────────────────────────────────────

class TestHealthEndpoint:
    def test_status_200(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_modelo_carregado(self, api_client):
        data = r = api_client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "ok"

    def test_ticker_correto(self, api_client):
        data = api_client.get("/health").json()
        assert data["ticker"] == "PETR4.SA"


class TestRootEndpoint:
    def test_status_200(self, api_client):
        r = api_client.get("/")
        assert r.status_code == 200

    def test_mensagem_presente(self, api_client):
        data = api_client.get("/").json()
        assert "message" in data


# ── Testes de Previsão ───────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predicao_basica_retorna_200(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 1}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_estrutura_da_resposta(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 1}
        data = api_client.post("/predict", json=payload).json()
        assert "ticker" in data
        assert "predictions" in data
        assert "days_ahead" in data
        assert "model_version" in data

    def test_ticker_petr4_na_resposta(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 1}
        data = api_client.post("/predict", json=payload).json()
        assert data["ticker"] == "PETR4.SA"

    def test_numero_de_previsoes_bate_com_days_ahead(self, api_client):
        for days in [1, 3, 5]:
            payload = {"prices": _prices(60), "days_ahead": days}
            data = api_client.post("/predict", json=payload).json()
            assert len(data["predictions"]) == days, (
                f"Esperado {days} previsões, obtido {len(data['predictions'])}"
            )

    def test_previsoes_sao_numeros(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 3}
        data = api_client.post("/predict", json=payload).json()
        for p in data["predictions"]:
            assert isinstance(p, float), f"Previsão deve ser float, obtido {type(p)}"

    def test_previsoes_em_range_plausivel(self, api_client):
        """Preços da PETR4 devem estar entre R$1 e R$200 num cenário normal."""
        payload = {"prices": _prices(90), "days_ahead": 5}
        data = api_client.post("/predict", json=payload).json()
        for p in data["predictions"]:
            assert 1.0 <= p <= 200.0, f"Preço fora do range plausível: {p}"

    def test_aceita_mais_de_60_precos(self, api_client):
        payload = {"prices": _prices(120), "days_ahead": 1}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_dias_maximo_30(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 30}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 200
        assert len(r.json()["predictions"]) == 30

    # Casos de erro ────────────────────────────────────────────────────────

    def test_menos_de_60_precos_retorna_422(self, api_client):
        payload = {"prices": _prices(30), "days_ahead": 1}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_days_ahead_zero_retorna_422(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 0}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_days_ahead_maior_30_retorna_422(self, api_client):
        payload = {"prices": _prices(60), "days_ahead": 31}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_lista_vazia_retorna_422(self, api_client):
        payload = {"prices": [], "days_ahead": 1}
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_payload_sem_prices_retorna_422(self, api_client):
        r = api_client.post("/predict", json={"days_ahead": 1})
        assert r.status_code == 422

    def test_payload_vazio_retorna_422(self, api_client):
        r = api_client.post("/predict", json={})
        assert r.status_code == 422


# ── Testes de Monitoramento ──────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_status_200(self, api_client):
        r = api_client.get("/metrics-summary")
        assert r.status_code == 200

    def test_campos_presentes(self, api_client):
        data = api_client.get("/metrics-summary").json()
        campos = ["uptime_seconds", "total_requests", "avg_response_time_ms",
                  "cpu_percent", "memory_mb"]
        for campo in campos:
            assert campo in data, f"Campo '{campo}' ausente na resposta"

    def test_uptime_positivo(self, api_client):
        data = api_client.get("/metrics-summary").json()
        assert data["uptime_seconds"] >= 0

    def test_total_requests_incrementa(self, api_client):
        antes = api_client.get("/metrics-summary").json()["total_requests"]
        api_client.get("/health")
        depois = api_client.get("/metrics-summary").json()["total_requests"]
        assert depois > antes

    def test_memoria_positiva(self, api_client):
        data = api_client.get("/metrics-summary").json()
        assert data["memory_mb"] > 0


# ── Testes de Modelo Não Carregado ───────────────────────────────────────────

class TestModelNaoCarregado:
    def test_retorna_503_sem_modelo(self, api_client):
        import api.main as main_module

        original_model = main_module.model
        main_module.model = None
        try:
            payload = {"prices": _prices(60), "days_ahead": 1}
            r = api_client.post("/predict", json=payload)
            assert r.status_code == 503
        finally:
            main_module.model = original_model

    def test_health_informa_modelo_nao_carregado(self, api_client):
        import api.main as main_module

        original_model = main_module.model
        main_module.model = None
        try:
            data = api_client.get("/health").json()
            assert data["model_loaded"] is False
            assert data["status"] != "ok"
        finally:
            main_module.model = original_model
