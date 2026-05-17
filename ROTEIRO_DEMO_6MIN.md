# ROTEIRO DEMO AO VIVO — 6 Minutos
# Tech Challenge Fase 4 | Sem slides — tudo no código e no terminal
# ============================================================
# ANTES DE GRAVAR — deixe tudo aberto e pronto:
#   ✅ VS Code aberto na pasta do projeto
#   ✅ Terminal aberto dentro do VS Code (Ctrl + ')
#   ✅ Navegador com http://localhost:8000/docs numa aba
#   ✅ Navegador com https://techchallengev4-production.up.railway.app/docs noutra aba
#   ✅ API local rodando: uvicorn api.main:app --reload
#   ✅ Payload copiado na área de transferência (no final desse arquivo)
# ============================================================


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCO 1 — INTRODUÇÃO + ESTRUTURA DO PROJETO  (1 min)
[ VS Code aberto — Explorer lateral visível ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Olá! Eu sou o Rafael, e esse é o Tech Challenge da Fase 4
da POSTECH Machine Learning Engineering.

O desafio foi criar um modelo LSTM para prever o preço de
fechamento da PETR4.SA e fazer o deploy completo em uma API
em produção. Vou mostrar tudo funcionando ao vivo.

[ Mostre o Explorer do VS Code com as pastas ]

O projeto está organizado assim: a pasta model tem o script de
treino e o modelo salvo. A pasta api tem toda a lógica da API
com FastAPI. A pasta monitoring tem o middleware de logs e
métricas. E a pasta tests tem os 66 testes automatizados.

Tudo containerizado com Docker para o deploy."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCO 2 — O MODELO LSTM  (1 min)
[ Abra model/train.py — mostre as partes principais ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Aqui está o script de treinamento.

[ Role até a função download_data — linha 33 ]

Primeiro baixamos os dados da PETR4.SA direto do Yahoo Finance
via yfinance — 1.738 pregões de 2018 a 2024.

[ Role até a função build_sequences — linha 44 ]

Aqui preparamos os dados em janelas de 60 dias. O modelo recebe
os últimos 60 pregões e aprende a prever o próximo preço.

[ Role até a função build_model — linha 53 ]

E aqui está a arquitetura LSTM: duas camadas recorrentes de
128 e 64 unidades, Dropout de 20% para evitar overfitting,
e uma saída com um único neurônio — o preço previsto.

O modelo já foi treinado e ficou com MAPE de 3.62% —
errando em média menos de 4% do preço real da ação.

[ Mostre os arquivos model/lstm_model.keras e model/scaler.pkl
  no Explorer ]

Esses são os arquivos gerados pelo treino: o modelo salvo
em formato .keras e o scaler que normaliza os dados."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCO 3 — A API  (1 min)
[ Abra api/main.py ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Aqui está a API construída com FastAPI.

[ Mostre a função lifespan — linha 24 ]

No startup da aplicação, o modelo e o scaler são carregados
do disco para a memória — prontos para inferência.

[ Mostre o endpoint predict — linha 60 ]

O endpoint /predict recebe uma lista de pelo menos 60 preços
históricos, normaliza com o scaler, alimenta o modelo e
devolve as previsões desnormalizadas em reais.

[ Abra monitoring/middleware.py ]

E aqui o middleware de monitoramento: ele intercepta cada
requisição, mede o tempo de resposta e registra tudo em log.
Esses dados são expostos para o Prometheus em /metrics."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCO 4 — TESTES AUTOMATIZADOS  (45 segundos)
[ Abra o terminal do VS Code ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Para garantir a qualidade, implementei 66 testes automatizados
cobrindo o pré-processamento, o modelo e todos os endpoints.

[ Digite e execute: ]

    python -m pytest

[ Aguarde rodar e mostre o resultado ]

66 testes passando, com 97% de cobertura de código na API
e 100% no middleware de monitoramento."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCO 5 — DEMO DA API LOCAL  (1 min 30 seg)
[ Navegador em http://localhost:8000/docs ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Agora vou mostrar a API funcionando.

[ GET /health → Try it out → Execute ]

O /health confirma que o modelo está carregado e a API está ativa.

[ GET /metrics-summary → Try it out → Execute ]

O /metrics-summary é o monitoramento em tempo real: uptime,
total de requisições, tempo médio de resposta, CPU e memória.
Esse é o requisito de escalabilidade e monitoramento do projeto.

[ POST /predict → Try it out → cole o payload → Execute ]

E o endpoint principal: o /predict. Estou enviando os últimos
60 preços reais de fechamento da PETR4 e pedindo a previsão
para os próximos 5 dias.

[ Mostre a resposta ]

Vejam: o modelo retornou os preços previstos em reais para os
próximos 5 pregões, junto com o ticker e a versão do modelo."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCO 6 — API EM PRODUÇÃO  (45 segundos)
[ Navegador em https://techchallengev4-production.up.railway.app/docs ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"A mesma API está deployada em produção no Railway com Docker.

[ Mostre a URL no navegador ]

Aqui está rodando na nuvem. Vou repetir o /predict para mostrar
que funciona em produção da mesma forma.

[ POST /predict → cole o payload → Execute ]

Mesmo resultado — modelo em produção, disponível publicamente.

[ Mostre o terminal com o log aparecendo ]

E no terminal podemos ver os logs registrando cada requisição:
método, rota, status e tempo de resposta em milissegundos."


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCERRAMENTO  (30 segundos)
[ Volte para o VS Code — Explorer visível ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Resumindo o que foi desenvolvido nesse projeto:

Coletamos 1.738 pregões da PETR4 com yfinance,
treinamos um LSTM que alcançou MAPE de 3.62%,
salvamos o modelo e buildamos uma API com FastAPI,
garantimos a qualidade com 66 testes automatizados,
e fizemos o deploy com Docker no Railway.

O código completo está no GitHub:
github.com/rafaelbuzato/TechChallengeV4

Obrigado!"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKLIST — ANTES DE GRAVAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [ ] API local rodando:  uvicorn api.main:app --reload
  [ ] Aba 1 no navegador: http://localhost:8000/docs
  [ ] Aba 2 no navegador: https://techchallengev4-production.up.railway.app/docs
  [ ] VS Code com model/train.py aberto
  [ ] VS Code com api/main.py aberto em outra aba
  [ ] VS Code com monitoring/middleware.py aberto em outra aba
  [ ] Terminal limpo dentro do VS Code
  [ ] Payload copiado na área de transferência (abaixo)
  [ ] Resolução da tela em 1080p
  [ ] Microfone testado


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAYLOAD — COPIE ANTES DE GRAVAR
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
