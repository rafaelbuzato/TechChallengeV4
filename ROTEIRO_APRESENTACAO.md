# ROTEIRO DE APRESENTAÇÃO — Tech Challenge Fase 4
# LSTM Stock Price Predictor — PETR4.SA
# Tempo estimado: 8 a 10 minutos
# ============================================================
# INSTRUÇÕES: As falas estão entre aspas. Siga as ações
# indicadas entre colchetes [ ] durante a gravação.
# ============================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 1 — CAPA
[ Mostre o slide de capa ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Olá! Meu nome é Rafael Buzato, e essa é a apresentação do
Tech Challenge da Fase 4 do curso de Machine Learning Engineering
da POSTECH.

O objetivo desse projeto foi desenvolver um modelo preditivo de
redes neurais LSTM — Long Short-Term Memory — para prever o preço
de fechamento de ações da bolsa de valores, e realizar a pipeline
completa de desenvolvimento: desde a coleta dos dados históricos
até o deploy do modelo em uma API RESTful em produção.

Para esse desafio, escolhi trabalhar com as ações da Petrobras,
o ticker PETR4.SA."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 2 — O DESAFIO
[ Avance para o slide 2 ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"O projeto seguiu cinco requisitos principais definidos no enunciado.

O primeiro foi a coleta e o pré-processamento dos dados. Para isso,
utilizei a biblioteca yfinance para baixar automaticamente os dados
históricos da PETR4.SA, cobrindo o período de janeiro de 2018 até
dezembro de 2024 — totalizando 1.738 pregões registrados.

O segundo foi o desenvolvimento do modelo LSTM, incluindo a
construção da rede neural, o treinamento e a avaliação com métricas
como MAE, RMSE e MAPE.

O terceiro foi o salvamento e exportação do modelo treinado, para
que ele pudesse ser carregado pela API em tempo de inferência.

O quarto foi o deploy do modelo em uma API RESTful, construída com
FastAPI, permitindo que qualquer sistema envie dados históricos e
receba previsões de preços futuros.

E o quinto foi a escalabilidade e o monitoramento, com rastreamento
de performance em produção, incluindo tempo de resposta e utilização
de recursos."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 3 — ARQUITETURA DO PROJETO
[ Avance para o slide 3 ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Antes de entrar nos detalhes técnicos, deixa eu mostrar como o
projeto está organizado.

A pasta 'model' contém o script de treinamento do LSTM, o modelo
salvo após o treino e o scaler responsável pela normalização dos dados.

A pasta 'api' contém toda a lógica da API RESTful construída com
FastAPI, incluindo os endpoints e os schemas de validação com Pydantic.

A pasta 'monitoring' contém um middleware customizado que registra
logs de cada requisição e coleta métricas de uso em tempo real,
como CPU e memória. Também inclui a configuração do Prometheus para
integração com dashboards Grafana em produção.

A pasta 'tests' contém 66 testes automatizados, cobrindo o
pré-processamento dos dados, o comportamento do modelo e todos os
endpoints da API.

E por fim, temos o Dockerfile e o docker-compose.yml, que permitem
subir toda a infraestrutura da aplicação com um único comando,
incluindo a API, o Prometheus e o Grafana."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 4 — MODELO LSTM — ARQUITETURA
[ Avance para o slide 4 ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Vamos agora falar sobre o coração do projeto: o modelo LSTM.

LSTM é um tipo especial de rede neural recorrente. Ao contrário das
redes tradicionais, o LSTM tem memória — ele consegue aprender
dependências de longo prazo em sequências de dados. Isso o torna
ideal para séries temporais financeiras, onde o preço de hoje
depende do comportamento dos dias anteriores.

Para o pré-processamento, os dados de fechamento foram normalizados
com MinMaxScaler entre 0 e 1, e organizados em janelas de 60 dias.
Ou seja: para cada previsão, o modelo recebe os últimos 60 pregões
como entrada e retorna o próximo preço de fechamento previsto.

A arquitetura do modelo foi construída em camadas:
Começamos com uma camada LSTM de 128 unidades com retorno de sequência,
seguida de um Dropout de 20% para reduzir overfitting.
Depois, uma segunda camada LSTM de 64 unidades,
mais um Dropout de 20%,
uma camada Dense com 32 neurônios e ativação ReLU,
e a camada de saída com um único neurônio — o preço previsto.

O dataset foi dividido em 80% para treino e 20% para teste.
Utilizamos o otimizador Adam com função de perda MSE, e o
EarlyStopping com paciência de 10 épocas para evitar overfitting.
O modelo convergiu em apenas 42 épocas."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 5 — RESULTADOS DO MODELO
[ Avance para o slide 5 ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Os resultados do modelo no conjunto de teste foram muito
satisfatórios.

O MAE — Mean Absolute Error — foi de 0.031, indicando um erro
absoluto médio muito baixo nas previsões normalizadas.

O RMSE — Root Mean Square Error — foi de 0.036, penalizando
mais os erros maiores, e ainda assim mantendo um valor excelente.

E a métrica mais importante para a interpretação de negócio:
o MAPE — Erro Percentual Absoluto Médio — foi de apenas 3.62%.

Isso significa que, em média, o modelo erra menos de 4% do preço
real da ação. Para séries temporais financeiras, que são altamente
voláteis e influenciadas por fatores externos, um erro abaixo de 5%
já é considerado um resultado muito bom.

O modelo foi avaliado sobre 348 amostras do conjunto de teste,
representando os 20% finais da série histórica."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 6 — API RESTFUL — FASTAPI
[ Avance para o slide 6 ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Após o treinamento, o modelo foi salvo em formato .keras e o
scaler em formato .pkl, para serem carregados pela API no momento
da inicialização.

A API foi desenvolvida com FastAPI, e possui cinco endpoints:

O GET /health verifica se o modelo está carregado e retorna o
status da aplicação — essencial para healthchecks em produção.

O POST /predict é o endpoint principal. O usuário envia uma lista
com pelo menos 60 preços históricos de fechamento e especifica
quantos dias futuros deseja prever, de 1 a 30. A API retorna as
previsões desnormalizadas em reais, o ticker e a versão do modelo.

O GET /metrics-summary retorna métricas de uso em tempo real:
uptime, total de requisições, tempo médio de resposta, uso de CPU
e memória RAM.

O GET /metrics expõe as métricas no formato Prometheus, para
integração com ferramentas de observabilidade.

E o GET /docs exibe a documentação interativa Swagger UI,
gerada automaticamente pelo FastAPI."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 7 — DEMO DA API
[ Abra o navegador em http://localhost:8000/docs ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Agora vamos ver a API funcionando na prática.

[ Clique em GET /health → Try it out → Execute ]

Aqui temos o endpoint de saúde. O retorno confirma que o modelo
está carregado, o status é 'ok' e o ticker é PETR4.SA.

[ Clique em GET /metrics-summary → Try it out → Execute ]

Agora as métricas de monitoramento. Podemos ver o tempo de
atividade da API, o total de requisições que foram processadas,
o tempo médio de resposta em milissegundos, o percentual de CPU
e o uso de memória RAM em tempo real.

Esse é o requisito de escalabilidade e monitoramento do projeto:
rastrear a performance do modelo em produção.

[ Clique em POST /predict → Try it out ]

E agora o endpoint de previsão. Vou inserir os últimos 60 preços
reais de fechamento da PETR4 e solicitar a previsão para os
próximos 5 dias úteis.

[ Cole o payload e clique em Execute ]

Vejam o resultado: o modelo retornou as previsões dos próximos
5 dias em reais, junto com o ticker, o número de dias previstos
e a versão do modelo em produção.

Esse é exatamente o comportamento esperado pelo projeto: o usuário
fornece dados históricos e recebe previsões de preços futuros."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 8 — TESTES AUTOMATIZADOS
[ Avance para o slide 8 — depois abra o terminal do VS Code ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Para garantir a qualidade e a confiabilidade do projeto,
implementamos 66 testes automatizados com pytest, organizados
em três arquivos.

O test_preprocessing.py testa toda a lógica de pré-processamento:
a normalização MinMax, a geração das sequências de 60 dias,
a divisão dos dados sem vazamento entre treino e teste,
e o cálculo correto das métricas MAE, RMSE e MAPE.

O test_model.py testa o comportamento do modelo: a inferência com
inputs corretos, o reshape dos dados para o formato 3D esperado
pelo LSTM, e a persistência do scaler em disco com joblib.

E o test_api.py testa todos os endpoints da API: respostas com
status correto, validação de payloads inválidos com erro 422,
comportamento quando o modelo não está carregado com erro 503,
e incremento correto das métricas de monitoramento.

[ Execute no terminal: python -m pytest ]

Vejam: todos os 66 testes passando. A cobertura de código é de
97% na API, 100% nos schemas e 100% no middleware de monitoramento."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 9 — DEPLOY E MONITORAMENTO
[ Avance para o slide 9 — depois abra a URL do Railway ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Falando sobre o deploy, toda a aplicação foi containerizada com
Docker. O Dockerfile define a imagem da API, e o docker-compose.yml
orquestra três serviços: a API, o Prometheus para coleta de métricas
e o Grafana para dashboards visuais.

Para o deploy em produção, utilizei a plataforma Railway, que
detectou automaticamente o Dockerfile e realizou o build da imagem.
A configuração suporta reinicialização automática em caso de falha
e uso de variável de porta dinâmica.

[ Abra a URL do Railway /docs ]

E aqui está a mesma API rodando em produção, na nuvem.

[ Repita o POST /predict ]

O modelo recebe os dados, processa e retorna as previsões —
com o mesmo comportamento do ambiente local, mas disponível
publicamente na internet.

O monitoramento em produção registra logs estruturados de cada
requisição, com método, rota, status e tempo de resposta.
Esses dados são expostos para o Prometheus e podem ser visualizados
no Grafana com dashboards em tempo real."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLIDE 10 — ENCERRAMENTO
[ Avance para o slide final ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Para fechar, vou resumir o que foi desenvolvido nesse projeto.

Realizamos a coleta e o pré-processamento automatizado de 1.738
pregões da PETR4.SA usando yfinance.

Desenvolvemos um modelo LSTM que alcançou um MAPE de 3.62%,
errando em média menos de 4% do preço real da ação.

Construímos uma API RESTful com FastAPI, com endpoints de previsão,
monitoramento e documentação automática via Swagger.

Implementamos 66 testes automatizados com 97% de cobertura de código,
garantindo a qualidade e a confiabilidade de toda a pipeline.

Containerizamos a aplicação com Docker e realizamos o deploy em
produção na plataforma Railway.

E configuramos o monitoramento completo com middleware de logs,
métricas em tempo real e integração com Prometheus e Grafana.

O código completo está disponível no GitHub em:
github.com/rafaelbuzato/TechChallengeV4

E a API está em produção e acessível pelo link do Railway.

Obrigado!"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAYLOAD PARA USAR NO DEMO AO VIVO (POST /predict)
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
