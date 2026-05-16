# GUIA DE EXECUÇÃO — Tech Challenge Fase 4
# LSTM Stock Price Predictor — PETR4.SA
# ============================================================
# Siga os passos na ordem. Cada etapa depende da anterior.
# ============================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 1 — PRÉ-REQUISITOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Certifique-se de ter instalado:
  ✅ Python 3.11 ou superior
  ✅ Git
  ✅ Docker Desktop (para deploy local)
  ✅ VS Code (recomendado)

Verifique as versões no terminal:

    python --version
    git --version
    docker --version


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 2 — CLONAR O REPOSITÓRIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    git clone https://github.com/rafaelbuzato/TechChallengeV4.git
    cd TechChallengeV4

✅ Resultado esperado: pasta do projeto aberta no terminal.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 3 — INSTALAR DEPENDÊNCIAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    pip install -r requirements.txt

⏳ Aguarde — o TensorFlow pode demorar alguns minutos.

✅ Resultado esperado: "Successfully installed..." sem erros.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 4 — TREINAR O MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    set PYTHONIOENCODING=utf-8
    python -m model.train

O script vai:
  → Baixar os dados da PETR4.SA via yfinance (2018–2024)
  → Normalizar e preparar as sequências de 60 dias
  → Treinar o modelo LSTM com EarlyStopping
  → Exibir as métricas MAE, RMSE e MAPE
  → Salvar o modelo em:  model/lstm_model.keras
  → Salvar o scaler em:  model/scaler.pkl
  → Salvar o gráfico em: data/prediction_plot.png

⏳ Tempo estimado: 3 a 8 minutos (depende da máquina).

✅ Resultado esperado:
    ── Test Metrics ──────────────────
      MAE  : ~0.03
      RMSE : ~0.03
      MAPE : ~3.6%
    Modelo salvo em model/lstm_model.keras


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 5 — RODAR OS TESTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python -m pytest

✅ Resultado esperado:
    66 passed in X.XXs
    Cobertura: api/main.py 97%, schemas.py 100%, middleware.py 100%

⚠️  Se algum teste falhar, verifique se o modelo foi treinado
    corretamente na etapa anterior.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 6 — RODAR A API LOCALMENTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    uvicorn api.main:app --reload

✅ Resultado esperado:
    INFO: Uvicorn running on http://127.0.0.1:8000
    Modelo e scaler carregados com sucesso.

Acesse no navegador:
  → Documentação: http://localhost:8000/docs
  → Health check: http://localhost:8000/health
  → Métricas:     http://localhost:8000/metrics-summary

Para parar a API: Ctrl + C


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 7 — TESTAR O ENDPOINT DE PREVISÃO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Com a API rodando, acesse http://localhost:8000/docs e:

  1. Clique em POST /predict
  2. Clique em "Try it out"
  3. Substitua o corpo da requisição pelo payload abaixo
  4. Clique em "Execute"

PAYLOAD:
{
  "prices": [
    30.77, 31.14, 31.06, 31.5,  30.87, 30.55, 30.91, 30.88, 30.96, 30.7,
    30.55, 30.32, 30.23, 29.76, 29.64, 29.27, 29.47, 29.68, 29.63, 29.56,
    29.43, 29.48, 29.08, 29.14, 29.05, 29.06, 29.15, 29.7,  29.76, 30.32,
    30.27, 30.59, 31.36, 31.03, 31.12, 32.36, 32.16, 32.12, 32.01, 31.68,
    31.93, 32.14, 32.43, 32.22, 32.54, 32.04, 32.87, 32.99, 33.32, 32.72,
    32.52, 32.38, 32.69, 31.85, 31.72, 31.45, 31.46, 31.7,  31.6,  32.07
  ],
  "days_ahead": 5
}

✅ Resultado esperado:
{
  "ticker": "PETR4.SA",
  "predictions": [32.XX, 32.XX, 32.XX, 32.XX, 32.XX],
  "days_ahead": 5,
  "model_version": "1.0.0"
}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 8 — RODAR COM DOCKER (OPCIONAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  Certifique-se que o Docker Desktop está aberto antes.
⚠️  O modelo (lstm_model.keras) precisa estar treinado (etapa 4).

    docker-compose up --build

Serviços disponíveis após o build:

  → API:        http://localhost:8000/docs
  → Prometheus: http://localhost:9090
  → Grafana:    http://localhost:3000
                Login: admin / Senha: admin

Para parar todos os serviços:

    docker-compose down

⏳ O primeiro build demora mais pois instala todas as dependências.
   Builds subsequentes são muito mais rápidos.

✅ Resultado esperado:
    petr4-lstm-api    | INFO: Uvicorn running on http://0.0.0.0:8000
    petr4-lstm-api    | Modelo e scaler carregados com sucesso.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ETAPA 9 — API EM PRODUÇÃO (RAILWAY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A API está disponível em produção no Railway.
Acesse pelo domínio gerado:

  → https://SEU-DOMINIO.up.railway.app/docs

Todos os endpoints funcionam da mesma forma que no ambiente local.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERÊNCIA RÁPIDA — COMANDOS PRINCIPAIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INSTALAR DEPENDÊNCIAS     pip install -r requirements.txt
  TREINAR MODELO            python -m model.train
  RODAR TESTES              python -m pytest
  SUBIR API LOCAL           uvicorn api.main:app --reload
  SUBIR COM DOCKER          docker-compose up --build
  PARAR DOCKER              docker-compose down
  ATUALIZAR REPOSITÓRIO     git pull origin main


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOLUÇÃO DE PROBLEMAS COMUNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROBLEMA: UnicodeEncodeError ao rodar o treino
SOLUÇÃO:  set PYTHONIOENCODING=utf-8  (antes do python -m model.train)

PROBLEMA: ModuleNotFoundError ao subir a API
SOLUÇÃO:  pip install -r requirements.txt

PROBLEMA: 503 ao chamar /predict
SOLUÇÃO:  O modelo não foi carregado. Verifique se lstm_model.keras
          e scaler.pkl existem na pasta model/

PROBLEMA: 422 ao chamar /predict
SOLUÇÃO:  Envie pelo menos 60 preços no campo "prices"

PROBLEMA: Porta 8000 já em uso
SOLUÇÃO:  uvicorn api.main:app --reload --port 8001

PROBLEMA: Docker não encontra o modelo
SOLUÇÃO:  Treine o modelo antes do docker-compose up --build
