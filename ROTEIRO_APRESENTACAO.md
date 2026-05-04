# Roteiro de Apresentação — Tech Challenge Fase 4
# PETR4.SA LSTM Stock Price Predictor
# Tempo estimado: 8 a 12 minutos

---

## ▶ PARTE 1 — INTRODUÇÃO (1 min)
> Câmera ou slide de capa

"Olá! Meu nome é Rafael, e essa é a apresentação do Tech Challenge da Fase 4
do curso de Machine Learning Engineering da POSTECH.

O desafio proposto foi desenvolver um modelo de redes neurais LSTM —
Long Short-Term Memory — para prever o preço de fechamento de ações
da bolsa de valores, e realizar toda a pipeline de desenvolvimento,
desde a coleta dos dados até o deploy do modelo em uma API em produção.

Para isso, escolhi trabalhar com as ações da Petrobras, o ticker PETR4.SA,
utilizando dados históricos de janeiro de 2018 até dezembro de 2024,
totalizando 1.738 registros de pregões."

---

## ▶ PARTE 2 — ARQUITETURA DO PROJETO (1-2 min)
> Mostre a pasta do projeto aberta no VS Code

"Antes de entrar no código, deixa eu mostrar como o projeto está organizado.

Temos cinco grandes blocos:

O primeiro é a pasta 'model', que contém o script de treinamento,
o modelo LSTM salvo após o treino, e o scaler que normaliza os dados.

O segundo é a pasta 'api', que contém a API RESTful construída com FastAPI,
com os endpoints de previsão, saúde e métricas.

O terceiro é a pasta 'monitoring', com o middleware que registra logs
de cada requisição e coleta métricas de uso como CPU e memória.
Também tem a configuração do Prometheus para monitoramento em produção.

O quarto é a pasta 'tests', com 66 testes automatizados cobrindo
pré-processamento, modelo e todos os endpoints da API.

E por fim, temos o Dockerfile e o docker-compose.yml,
que permitem subir toda a infraestrutura com um único comando."

---

## ▶ PARTE 3 — O MODELO LSTM (2 min)
> Mostre o arquivo model/train.py ou o notebook

"Vamos falar sobre o modelo.

LSTM é um tipo especial de rede neural recorrente, muito eficiente
para dados sequenciais e séries temporais — como é o caso de preços de ações,
onde o valor de hoje depende dos valores anteriores.

O modelo foi construído com a seguinte arquitetura:
Começamos com uma camada LSTM de 128 unidades com retorno de sequência,
seguida de um Dropout de 20% para evitar overfitting.
Depois, uma segunda camada LSTM de 64 unidades,
mais um Dropout de 20%,
uma camada Dense com 32 neurônios e ativação ReLU,
e a camada de saída com 1 neurônio — o preço previsto.

Para treinar, usamos os dados normalizados com MinMaxScaler,
com janelas de 60 dias como entrada — ou seja, o modelo olha
para os últimos 60 pregões para prever o próximo.

O conjunto foi dividido em 80% para treino e 20% para teste.
Usamos EarlyStopping, e o modelo convergiu em apenas 42 épocas.

Os resultados foram excelentes:
- MAE de 0.031
- RMSE de 0.036
- e um MAPE de apenas 3.62% —
  isso significa que o modelo erra em média menos de 4% do preço real.

[MOSTRAR O GRÁFICO data/prediction_plot.png]

Aqui podemos ver o gráfico com os valores reais em azul
e os valores previstos em laranja tracejado.
O modelo acompanha muito bem a tendência dos preços."

---

## ▶ PARTE 4 — DEMO DA API LOCAL (2 min)
> Abra http://localhost:8000/docs no navegador

"Agora vou mostrar a API funcionando localmente.

A API foi desenvolvida com FastAPI, que gera automaticamente
essa documentação interativa que vocês estão vendo — o Swagger UI.

Temos quatro endpoints principais:

[CLIQUE EM GET /health → Try it out → Execute]
Primeiro o endpoint de saúde. Ele confirma que o modelo está carregado
e qual ticker está sendo servido.

[CLIQUE EM GET /metrics-summary → Try it out → Execute]
Aqui temos as métricas em tempo real: tempo de atividade da API,
total de requisições processadas, tempo médio de resposta,
percentual de CPU e uso de memória RAM.

[CLIQUE EM POST /predict → Try it out]
E agora o endpoint mais importante — o de previsão.

Vou passar os últimos 60 preços reais de fechamento da PETR4
e pedir a previsão para os próximos 5 dias úteis.

[COLE O PAYLOAD E CLIQUE EM EXECUTE]

Vejam a resposta: o modelo retornou as previsões para os próximos 5 dias,
em reais, junto com o ticker, o número de dias previstos
e a versão do modelo."

---

## ▶ PARTE 5 — DEMO DA API EM PRODUÇÃO (1 min)
> Abra a URL do Railway no navegador

"Além de funcionar localmente, a API está deployada em produção
na plataforma Railway, usando o Docker que configuramos.

[ACESSE A URL DO RAILWAY /docs]

Aqui está a mesma API rodando em nuvem, com a mesma documentação.
Vou repetir a chamada de previsão para demonstrar que funciona
em ambiente de produção.

[REPITA O POST /predict]

Perfeito — mesmos resultados, rodando em produção."

---

## ▶ PARTE 6 — TESTES AUTOMATIZADOS (1 min)
> Abra o terminal do VS Code

"Para garantir a qualidade do projeto, implementamos 66 testes automatizados
organizados em três arquivos:

'test_preprocessing.py' testa toda a lógica de normalização dos dados,
geração das sequências e validação das métricas.

'test_model.py' testa a inferência do modelo, o reshape dos inputs
e a persistência do scaler em disco.

'test_api.py' testa todos os endpoints — respostas corretas,
validações de entrada e comportamento quando o modelo não está carregado.

[RODE: python -m pytest]

Todos os 66 testes passando. A cobertura de código é de 97%
nos módulos da API e 100% no middleware de monitoramento."

---

## ▶ PARTE 7 — ENCERRAMENTO (30 seg)
> Câmera ou slide final

"Para resumir, nesse projeto desenvolvemos:
uma coleta e pré-processamento automatizado dos dados da PETR4,
um modelo LSTM com MAPE de 3.62%,
uma API RESTful com FastAPI com endpoints de previsão e monitoramento,
testes automatizados com 97% de cobertura,
e deploy em produção com Docker e Railway.

O código completo está disponível no GitHub em:
github.com/rafaelbuzato/TechChallengeV4

Obrigado!"

---

## 📋 PAYLOAD PARA USAR NO DEMO
Cole isso no corpo do POST /predict:

{
  "prices": [
    30.77, 31.14, 31.06, 31.5, 30.87, 30.55, 30.91, 30.88, 30.96, 30.7,
    30.55, 30.32, 30.23, 29.76, 29.64, 29.27, 29.47, 29.68, 29.63, 29.56,
    29.43, 29.48, 29.08, 29.14, 29.05, 29.06, 29.15, 29.7, 29.76, 30.32,
    30.27, 30.59, 31.36, 31.03, 31.12, 32.36, 32.16, 32.12, 32.01, 31.68,
    31.93, 32.14, 32.43, 32.22, 32.54, 32.04, 32.87, 32.99, 33.32, 32.72,
    32.52, 32.38, 32.69, 31.85, 31.72, 31.45, 31.46, 31.7, 31.6, 32.07
  ],
  "days_ahead": 5
}
