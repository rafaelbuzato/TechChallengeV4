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
│   └── scaler.pkl               # Scaler MinMax (gerado após treino)
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
- Salvar `model/lstm_model.keras` e `model/scaler.pkl`
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

| Método | Rota               | Descrição                            |
|--------|--------------------|--------------------------------------|
| GET    | `/`                | Página inicial                       |
| GET    | `/health`          | Status da API e do modelo            |
| POST   | `/predict`         | Previsão de preços futuros           |
| GET    | `/metrics-summary` | Métricas de uso (tempo, CPU, RAM)    |
| GET    | `/metrics`         | Métricas Prometheus                  |
| GET    | `/docs`            | Documentação Swagger UI              |

### Exemplo de requisição `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prices": [<lista com 60+ preços históricos>],
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
