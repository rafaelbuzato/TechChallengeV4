# ROTEIRO — Versão 6 Minutos
# Tech Challenge Fase 4 | PETR4.SA LSTM Predictor
# ============================================================
# Cada bloco tem o tempo estimado entre parênteses.
# Fale em ritmo normal — sem pressa, sem enrolar.
# ============================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 1 — CAPA  (30 segundos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Olá! Eu sou Rafael Buzato, e esse é o Tech Challenge da Fase 4
da POSTECH Machine Learning Engineering.

O desafio foi criar um modelo LSTM para prever o preço de
fechamento de ações — e fazer o deploy completo em uma API
em produção. Escolhi trabalhar com a Petrobras, ticker PETR4.SA."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 2 — O DESAFIO  (30 segundos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"O projeto cobriu os cinco requisitos do enunciado: coleta e
pré-processamento dos dados com yfinance — 1.738 pregões de 2018
a 2024 —, desenvolvimento e avaliação do modelo LSTM, salvamento
do modelo para inferência, deploy em API RESTful e monitoramento
em produção."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDES 3 e 4 — ARQUITETURA + MODELO  (1 minuto)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"O projeto está organizado em quatro módulos: model para o treino
e o modelo salvo, api com a lógica FastAPI, monitoring com logs e
métricas, e tests com os testes automatizados. Tudo containerizado
com Docker.

O modelo LSTM usa janelas de 60 dias como entrada para prever o
próximo preço de fechamento. A arquitetura tem duas camadas LSTM —
128 e 64 unidades — com Dropout de 20% entre elas para evitar
overfitting, finalizando em uma camada densa de saída.

Os dados foram normalizados com MinMaxScaler, divididos em 80%
treino e 20% teste, e o treinamento usou EarlyStopping —
convergindo em 42 épocas."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 5 — RESULTADOS  (30 segundos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Os resultados no conjunto de teste foram excelentes.
MAE de 0.031, RMSE de 0.036, e o mais importante:
MAPE de apenas 3.62% — o modelo erra em média menos de 4%
do preço real. Para séries financeiras, isso é um resultado
muito sólido."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDES 6 e 7 — API + DEMO AO VIVO  (2 minutos)
[ Abra http://localhost:8000/docs ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"A API foi construída com FastAPI e tem cinco endpoints.
Vou demonstrar os principais agora.

[ GET /health → Try it out → Execute ]
O /health confirma que o modelo está carregado e a API está ativa.

[ GET /metrics-summary → Try it out → Execute ]
O /metrics-summary mostra monitoramento em tempo real: uptime,
total de requisições, tempo médio de resposta, CPU e memória.
Esse é o requisito de escalabilidade e monitoramento do projeto.

[ POST /predict → Try it out → cole o payload → Execute ]
E o endpoint principal: o /predict. Estou enviando os últimos
60 preços reais de fechamento da PETR4 e pedindo previsão para
5 dias. Vejam a resposta — o modelo retornou os preços previstos
em reais, com o ticker e a versão do modelo.

[ Abra a URL do Railway ]
E aqui está a mesma API rodando em produção no Railway —
mesma chamada, mesmo resultado, disponível na internet."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 8 — TESTES  (30 segundos)
[ Abra o terminal do VS Code ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Para garantir a qualidade, o projeto tem 66 testes automatizados
cobrindo pré-processamento, modelo e todos os endpoints da API.

[ python -m pytest ]

66 testes passando, com 97% de cobertura de código."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 10 — ENCERRAMENTO  (30 segundos)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Resumindo: coletamos 1.738 pregões da PETR4, treinamos um LSTM
com MAPE de 3.62%, deployamos uma API com FastAPI e Docker no
Railway, e garantimos a qualidade com 66 testes automatizados.

Código no GitHub: github.com/rafaelbuzato/TechChallengeV4.
Obrigado!"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAYLOAD — POST /predict
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
