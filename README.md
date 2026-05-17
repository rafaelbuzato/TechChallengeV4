# PETR4.SA — LSTM Stock Price Predictor

**Tech Challenge Fase 4 | POSTECH Machine Learning Engineering**

Modelo preditivo LSTM para prever o preço de fechamento da PETR4.SA (Petrobras), com API RESTful e infraestrutura Docker para deploy.

---

## Estrutura do Projeto

```
TechChallengeV4/
├── data/                        # Dados históricos baixados e gráficos gerados
├── model/
│   ├── train.py                 # Script de treinamento do modelo LSTM
│   ├── lstm_model.keras         # Modelo treinado (gerado após treino)
│   ├── scaler.pkl               # Scaler MinMax (gerado após treino)
│   └── metrics.json             # Métricas e metadados do último treino
├── api/
│   ├── main.py                  # API FastAPI
│   └── schemas.py               # Modelos Pydantic
├── monitoring/
│   ├── middleware.py            # Middleware de monitoramento
│   └── prometheus.yml           # Configuração do Prometheus
├── notebooks/
│   └── eda_training.ipynb       # EDA + treinamento interativo
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Requisitos

- Python 3.11+
- Docker e Docker Compose

---

## Passo a Passo

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python -m model.train
```

O script irá:
- Baixar dados da PETR4.SA via `yfinance` (2018–2024)
- Treinar o modelo LSTM
- Salvar `model/lstm_model.keras`, `model/scaler.pkl` e `model/metrics.json`
- Exibir métricas MAE, RMSE e MAPE
- Gerar gráfico em `data/prediction_plot.png`

### 3. Rodar a API localmente

```bash
uvicorn api.main:app --reload
```

Acesse a documentação interativa em: `http://localhost:8000/docs`

### 4. Rodar com Docker

```bash
# Build e start de todos os serviços
docker-compose up --build

# Apenas a API
docker-compose up api
```

Serviços disponíveis:

| Serviço    | URL                          |
|------------|------------------------------|
| API        | http://localhost:8000/docs   |
| Prometheus | http://localhost:9090        |
| Grafana    | http://localhost:3000        |

> Grafana: usuário `admin`, senha `admin`

---

## Endpoints da API

| Método | Rota               | Descrição                                          |
|--------|--------------------|----------------------------------------------------|
| GET    | `/`                | Página inicial                                     |
| GET    | `/health`          | Status da API e do modelo                          |
| GET    | `/model-metrics`   | Métricas e metadados do modelo treinado            |
| POST   | `/predict`         | Previsão de preços futuros                         |
| GET    | `/metrics-summary` | Métricas de uso em tempo real (tempo, CPU, RAM)    |
| GET    | `/metrics`         | Métricas Prometheus (scraping)                     |
| GET    | `/docs`            | Documentação Swagger UI                            |

---

### `GET /model-metrics`

Retorna os resultados do treinamento e metadados completos do modelo.

```bash
curl http://localhost:8000/model-metrics
```

Resposta:

```json
{
  "ticker": "PETR4.SA",
  "model_version": "1.0.0",
  "trained_at": "2026-05-17 17:00:00",
  "train_period": "01/01/2018 a 31/12/2024",
  "architecture": "LSTM(128) → Dropout(20%) → LSTM(64) → Dropout(20%) → Dense(32, relu) → Dense(1)",
  "optimizer": "Adam",
  "loss_function": "Mean Squared Error",
  "total_records": 1738,
  "train_samples": 1330,
  "test_samples": 348,
  "sequence_length": 60,
  "epochs_executed": 42,
  "mae": 0.0309,
  "rmse": 0.0356,
  "mape": 3.62,
  "mape_rating": "Excelente — erro médio abaixo de 5%"
}
```

---

### `POST /predict`

Recebe uma lista com pelo menos 60 preços históricos de fechamento e retorna a previsão para os próximos N dias.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [30.77, 31.14, 31.06, ...],
    "days_ahead": 5
  }'
```

Resposta:

```json
{
  "ticker": "PETR4.SA",
  "predictions": [38.12, 38.45, 37.98, 38.71, 39.02],
  "days_ahead": 5,
  "model_version": "1.0.0"
}
```

| Campo        | Tipo    | Descrição                                    |
|--------------|---------|----------------------------------------------|
| `prices`     | float[] | Mínimo de 60 preços históricos (ordem cronológica) |
| `days_ahead` | int     | Dias a prever — entre 1 e 30 (padrão: 1)    |

---

## Modelo LSTM

| Camada     | Configuração                |
|------------|-----------------------------|
| LSTM       | 128 unidades, return_seq=True |
| Dropout    | 20%                         |
| LSTM       | 64 unidades                 |
| Dropout    | 20%                         |
| Dense      | 32 unidades, ReLU           |
| Dense      | 1 unidade (saída)           |

- **Janela de entrada:** 60 dias
- **Otimizador:** Adam
- **Loss:** MSE
- **Early Stopping:** paciência de 10 épocas

---

## Monitoramento

- **Logs estruturados** de cada requisição (método, rota, status, tempo)
- **Endpoint `/metrics-summary`** com uptime, total de requests, tempo médio de resposta, CPU e RAM
- **Prometheus + Grafana** via Docker Compose para dashboards em produção

---

## Tecnologias

- Python 3.11
- TensorFlow / Keras
- FastAPI + Uvicorn
- yfinance
- scikit-learn
- Docker + Docker Compose
- Prometheus + Grafana
