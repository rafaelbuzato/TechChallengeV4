const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Tech Challenge Fase 4 - LSTM PETR4";

// ── Paleta de cores ────────────────────────────────────────────────────────
const C = {
  bg:       "0D1117",   // fundo principal (quase preto)
  bgLight:  "161B22",   // fundo cards
  red:      "E63946",   // vermelho destaque
  blue:     "58A6FF",   // azul elétrico
  white:    "F0F6FC",   // texto principal
  gray:     "8B949E",   // texto secundário
  green:    "3FB950",   // verde sucesso
  yellow:   "F0E68C",   // amarelo destaque
  darkCard: "21262D",   // card escuro
};

const FONT = "Calibri";

// ── Helper: barra de título colorida no topo ──────────────────────────────
function addTopBar(slide, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.08,
    fill: { color: color || C.red },
    line: { color: color || C.red }
  });
}

// ── Helper: título padrão dos slides ──────────────────────────────────────
function addTitle(slide, text, color) {
  slide.addText(text, {
    x: 0.5, y: 0.18, w: 9, h: 0.65,
    fontSize: 28, bold: true,
    fontFace: FONT, color: color || C.white,
    margin: 0
  });
  // linha separadora sutil
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.85, w: 9, h: 0.025,
    fill: { color: C.blue, transparency: 40 },
    line: { color: C.blue, transparency: 40 }
  });
}

// ── Helper: card de fundo ─────────────────────────────────────────────────
function addCard(slide, x, y, w, h, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: color || C.bgLight },
    line: { color: C.gray, transparency: 70 }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — CAPA
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };

  // Barra lateral esquerda vermelha
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.2, h: 5.625,
    fill: { color: C.red }, line: { color: C.red }
  });

  // Barra lateral direita azul
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 9.8, y: 0, w: 0.2, h: 5.625,
    fill: { color: C.blue }, line: { color: C.blue }
  });

  // Título principal
  slide.addText("Tech Challenge — Fase 4", {
    x: 0.5, y: 1.2, w: 9, h: 1.0,
    fontSize: 44, bold: true, fontFace: FONT,
    color: C.white, align: "center"
  });

  // Subtítulo
  slide.addText("LSTM Stock Price Predictor", {
    x: 0.5, y: 2.3, w: 9, h: 0.7,
    fontSize: 28, bold: false, fontFace: FONT,
    color: C.blue, align: "center"
  });

  // Linha decorativa
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.0, y: 3.1, w: 4, h: 0.04,
    fill: { color: C.red }, line: { color: C.red }
  });

  // Empresa
  slide.addText("PETR4.SA — Petrobras", {
    x: 0.5, y: 3.25, w: 9, h: 0.5,
    fontSize: 20, fontFace: FONT,
    color: C.yellow, align: "center"
  });

  // Rodapé
  slide.addText("Machine Learning Engineering  |  POSTECH", {
    x: 0.5, y: 4.8, w: 9, h: 0.4,
    fontSize: 14, fontFace: FONT,
    color: C.gray, align: "center"
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — O DESAFIO
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.red);
  addTitle(slide, "O Desafio");

  const bullets = [
    "Criar um modelo preditivo de redes neurais LSTM",
    "Prever o preço de fechamento de ações da bolsa",
    "Empresa escolhida: Petrobras (PETR4.SA)",
    "Pipeline completa: dados → modelo → API → produção",
    "Dados: Janeiro/2018 a Dezembro/2024 — 1.738 pregões",
  ];

  addCard(slide, 0.5, 1.0, 9, 4.2);

  slide.addText(
    bullets.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i < bullets.length - 1 }
    })),
    {
      x: 0.8, y: 1.15, w: 8.4, h: 3.8,
      fontSize: 18, fontFace: FONT, color: C.white,
      paraSpaceAfter: 10, valign: "middle"
    }
  );

  // Badge lateral
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.0, w: 0.1, h: 4.2,
    fill: { color: C.red }, line: { color: C.red }
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — ARQUITETURA DO PROJETO
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.blue);
  addTitle(slide, "Arquitetura do Projeto", C.white);

  const items = [
    { dir: "model/",            desc: "Treinamento LSTM + modelo salvo",    color: C.red },
    { dir: "api/",              desc: "API RESTful com FastAPI",              color: C.blue },
    { dir: "monitoring/",       desc: "Middleware + Prometheus + Grafana",    color: C.green },
    { dir: "tests/",            desc: "66 testes automatizados",              color: C.yellow },
    { dir: "Dockerfile +\ndocker-compose.yml", desc: "Deploy containerizado", color: C.gray },
  ];

  items.forEach((item, i) => {
    const y = 1.05 + i * 0.88;
    addCard(slide, 0.4, y, 9.2, 0.76, C.darkCard);

    // Barra colorida esquerda
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.4, y, w: 0.12, h: 0.76,
      fill: { color: item.color }, line: { color: item.color }
    });

    // Nome do diretório
    slide.addText(item.dir, {
      x: 0.65, y: y + 0.1, w: 2.5, h: 0.55,
      fontSize: 15, bold: true, fontFace: "Consolas",
      color: item.color, valign: "middle", margin: 0
    });

    // Seta
    slide.addText("→", {
      x: 3.1, y: y + 0.1, w: 0.4, h: 0.55,
      fontSize: 16, fontFace: FONT, color: C.gray,
      valign: "middle", align: "center", margin: 0
    });

    // Descrição
    slide.addText(item.desc, {
      x: 3.5, y: y + 0.1, w: 5.9, h: 0.55,
      fontSize: 15, fontFace: FONT, color: C.white,
      valign: "middle", margin: 0
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — MODELO LSTM
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.red);
  addTitle(slide, "Modelo LSTM — Arquitetura");

  // Coluna esquerda: arquitetura
  addCard(slide, 0.4, 1.0, 4.5, 4.2, C.darkCard);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.0, w: 0.1, h: 4.2,
    fill: { color: C.red }, line: { color: C.red }
  });
  slide.addText("Arquitetura", {
    x: 0.65, y: 1.1, w: 4.0, h: 0.45,
    fontSize: 16, bold: true, fontFace: FONT, color: C.red, margin: 0
  });

  const layers = [
    { text: "LSTM 128 unidades", detail: "return_sequences=True" },
    { text: "Dropout 20%",       detail: "" },
    { text: "LSTM 64 unidades",  detail: "" },
    { text: "Dropout 20%",       detail: "" },
    { text: "Dense 32 (ReLU)",   detail: "" },
    { text: "Dense 1",           detail: "Saída — preço previsto" },
  ];

  layers.forEach((l, i) => {
    const y = 1.65 + i * 0.56;
    const isOutput = i === layers.length - 1;
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.75, y, w: 3.9, h: 0.42,
      fill: { color: isOutput ? C.red : C.bgLight },
      line: { color: isOutput ? C.red : C.blue, transparency: isOutput ? 0 : 30 }
    });
    slide.addText(
      l.detail
        ? [{ text: l.text + "  ", options: { bold: true } }, { text: l.detail, options: { color: C.gray, fontSize: 12 } }]
        : l.text,
      {
        x: 0.85, y: y + 0.04, w: 3.7, h: 0.34,
        fontSize: 13, fontFace: FONT, color: C.white, valign: "middle", margin: 0
      }
    );
  });

  // Coluna direita: configurações
  addCard(slide, 5.1, 1.0, 4.5, 4.2, C.darkCard);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.0, w: 0.1, h: 4.2,
    fill: { color: C.blue }, line: { color: C.blue }
  });
  slide.addText("Configurações", {
    x: 5.35, y: 1.1, w: 4.0, h: 0.45,
    fontSize: 16, bold: true, fontFace: FONT, color: C.blue, margin: 0
  });

  const configs = [
    ["Janela de entrada", "60 dias"],
    ["Otimizador", "Adam"],
    ["Função de perda", "MSE"],
    ["EarlyStopping", "Paciência: 10 épocas"],
    ["Split dos dados", "80% treino / 20% teste"],
    ["Épocas executadas", "42 (parada antecipada)"],
  ];

  configs.forEach(([label, value], i) => {
    const y = 1.65 + i * 0.56;
    slide.addText([
      { text: label + ":  ", options: { color: C.gray } },
      { text: value, options: { bold: true, color: C.white } }
    ], {
      x: 5.35, y: y + 0.04, w: 4.1, h: 0.48,
      fontSize: 13, fontFace: FONT, valign: "middle", margin: 0
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — RESULTADOS
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.green);
  addTitle(slide, "Resultados do Modelo", C.white);

  // 3 cards de métricas
  const metrics = [
    { label: "MAE",  value: "0.031", color: C.blue,  sub: "Mean Absolute Error" },
    { label: "RMSE", value: "0.036", color: C.yellow, sub: "Root Mean Square Error" },
    { label: "MAPE", value: "3.62%", color: C.green,  sub: "Erro Médio Percentual ★" },
  ];

  metrics.forEach((m, i) => {
    const x = 0.4 + i * 3.1;
    addCard(slide, x, 1.1, 2.9, 2.5, C.darkCard);
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.1, w: 2.9, h: 0.12,
      fill: { color: m.color }, line: { color: m.color }
    });
    slide.addText(m.label, {
      x: x + 0.1, y: 1.35, w: 2.7, h: 0.5,
      fontSize: 18, bold: true, fontFace: FONT,
      color: m.color, align: "center", margin: 0
    });
    slide.addText(m.value, {
      x: x + 0.1, y: 1.9, w: 2.7, h: 0.9,
      fontSize: 42, bold: true, fontFace: FONT,
      color: C.white, align: "center", margin: 0
    });
    slide.addText(m.sub, {
      x: x + 0.1, y: 2.85, w: 2.7, h: 0.55,
      fontSize: 11, fontFace: FONT,
      color: C.gray, align: "center", margin: 0
    });
  });

  // Banner de destaque
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 3.8, w: 9.2, h: 0.72,
    fill: { color: C.green, transparency: 80 },
    line: { color: C.green, transparency: 50 }
  });
  slide.addText("Modelo convergiu em 42 épocas  •  Erro médio inferior a 4% do preço real", {
    x: 0.5, y: 3.87, w: 9.0, h: 0.58,
    fontSize: 16, fontFace: FONT, color: C.white,
    align: "center", bold: true, margin: 0
  });

  // Rodapé
  slide.addText("Conjunto de teste: 348 amostras (20% dos dados)", {
    x: 0.4, y: 4.65, w: 9.2, h: 0.35,
    fontSize: 12, fontFace: FONT, color: C.gray, align: "center", margin: 0
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — API FASTAPI
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.blue);
  addTitle(slide, "API RESTful — FastAPI");

  const endpoints = [
    { method: "GET",  path: "/health",          desc: "Status do modelo e da API",             mcolor: C.green },
    { method: "POST", path: "/predict",          desc: "Previsão de preços futuros (1–30 dias)", mcolor: C.blue },
    { method: "GET",  path: "/metrics-summary",  desc: "CPU, RAM e tempo de resposta",           mcolor: C.yellow },
    { method: "GET",  path: "/metrics",           desc: "Métricas para Prometheus",               mcolor: C.yellow },
    { method: "GET",  path: "/docs",              desc: "Documentação Swagger UI automática",      mcolor: C.gray },
  ];

  endpoints.forEach((ep, i) => {
    const y = 1.05 + i * 0.84;
    addCard(slide, 0.4, y, 9.2, 0.72, C.darkCard);

    // Badge do método
    slide.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: y + 0.12, w: 0.85, h: 0.48,
      fill: { color: ep.mcolor }, line: { color: ep.mcolor }
    });
    slide.addText(ep.method, {
      x: 0.5, y: y + 0.12, w: 0.85, h: 0.48,
      fontSize: 13, bold: true, fontFace: "Consolas",
      color: C.bg, align: "center", valign: "middle", margin: 0
    });

    // Path
    slide.addText(ep.path, {
      x: 1.5, y: y + 0.12, w: 2.8, h: 0.48,
      fontSize: 14, bold: true, fontFace: "Consolas",
      color: C.white, valign: "middle", margin: 0
    });

    // Seta
    slide.addText("→", {
      x: 4.3, y: y + 0.12, w: 0.4, h: 0.48,
      fontSize: 14, fontFace: FONT, color: C.gray,
      align: "center", valign: "middle", margin: 0
    });

    // Descrição
    slide.addText(ep.desc, {
      x: 4.7, y: y + 0.12, w: 4.7, h: 0.48,
      fontSize: 13, fontFace: FONT, color: C.gray,
      valign: "middle", margin: 0
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — DEMO DA PREVISÃO
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.blue);
  addTitle(slide, "Exemplo de Previsão");

  // Coluna esquerda — entrada
  addCard(slide, 0.4, 1.05, 4.4, 4.2, C.darkCard);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.05, w: 0.1, h: 4.2,
    fill: { color: C.blue }, line: { color: C.blue }
  });
  slide.addText("ENTRADA", {
    x: 0.65, y: 1.15, w: 3.9, h: 0.4,
    fontSize: 13, bold: true, fontFace: FONT, color: C.blue, margin: 0
  });
  slide.addText([
    { text: "prices", options: { color: C.blue, bold: true } },
    { text: ": 60 preços históricos\nde fechamento da PETR4", options: { color: C.white } },
    { text: "\n\n" },
    { text: "days_ahead", options: { color: C.blue, bold: true } },
    { text: ": 5", options: { color: C.white } },
    { text: "\n\n" },
    { text: "Exemplo dos últimos preços:", options: { color: C.gray, fontSize: 12 } },
    { text: "\n30.77, 31.14, 31.06, ...\n31.7, 31.6, 32.07", options: { color: C.yellow, fontSize: 12, fontFace: "Consolas" } },
  ], {
    x: 0.65, y: 1.65, w: 3.9, h: 3.3,
    fontSize: 14, fontFace: FONT, valign: "top", margin: 0
  });

  // Coluna direita — saída
  addCard(slide, 5.1, 1.05, 4.5, 4.2, C.darkCard);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.05, w: 0.1, h: 4.2,
    fill: { color: C.green }, line: { color: C.green }
  });
  slide.addText("SAÍDA", {
    x: 5.35, y: 1.15, w: 4.0, h: 0.4,
    fontSize: 13, bold: true, fontFace: FONT, color: C.green, margin: 0
  });

  // JSON de resposta
  slide.addText(
`{
  "ticker": "PETR4.SA",
  "predictions": [
    32.15,
    32.21,
    32.18,
    32.24,
    32.29
  ],
  "days_ahead": 5,
  "model_version": "1.0.0"
}`,
    {
      x: 5.35, y: 1.65, w: 4.1, h: 3.3,
      fontSize: 12, fontFace: "Consolas", color: C.white,
      valign: "top", margin: 0
    }
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — TESTES
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.green);
  addTitle(slide, "Testes Automatizados");

  // Badge 66 testes
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 7.2, y: 0.9, w: 2.3, h: 0.85,
    fill: { color: C.green }, line: { color: C.green }
  });
  slide.addText("66 testes ✓", {
    x: 7.2, y: 0.9, w: 2.3, h: 0.85,
    fontSize: 18, bold: true, fontFace: FONT,
    color: C.bg, align: "center", valign: "middle", margin: 0
  });

  // 3 cards de arquivos de teste
  const testFiles = [
    {
      name: "test_preprocessing.py",
      items: ["Normalização MinMax", "Geração de sequências", "Split treino/teste", "Cálculo MAE/RMSE/MAPE"],
      color: C.blue
    },
    {
      name: "test_model.py",
      items: ["Inferência do modelo", "Reshape de inputs", "Persistência do scaler", "Hiperparâmetros"],
      color: C.yellow
    },
    {
      name: "test_api.py",
      items: ["Todos os endpoints", "Validações de entrada", "Erros 422 e 503", "Modelo não carregado"],
      color: C.red
    },
  ];

  testFiles.forEach((tf, i) => {
    const y = 1.95;
    const x = 0.35 + i * 3.1;
    addCard(slide, x, y, 2.95, 2.5, C.darkCard);
    slide.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.95, h: 0.1,
      fill: { color: tf.color }, line: { color: tf.color }
    });
    slide.addText(tf.name, {
      x: x + 0.12, y: y + 0.15, w: 2.7, h: 0.45,
      fontSize: 11, bold: true, fontFace: "Consolas",
      color: tf.color, margin: 0
    });
    slide.addText(
      tf.items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < tf.items.length - 1 } })),
      {
        x: x + 0.12, y: y + 0.65, w: 2.7, h: 1.7,
        fontSize: 12, fontFace: FONT, color: C.white,
        paraSpaceAfter: 4, margin: 0
      }
    );
  });

  // Cobertura de código
  addCard(slide, 0.35, 4.6, 9.3, 0.75, C.darkCard);
  slide.addText("Cobertura de código:", {
    x: 0.55, y: 4.68, w: 2.2, h: 0.55,
    fontSize: 13, bold: true, fontFace: FONT, color: C.white, valign: "middle", margin: 0
  });

  const cov = [
    { file: "api/main.py", pct: "97%", color: C.blue },
    { file: "api/schemas.py", pct: "100%", color: C.green },
    { file: "monitoring/middleware.py", pct: "100%", color: C.green },
  ];
  cov.forEach((c, i) => {
    slide.addText([
      { text: c.file + "  ", options: { color: C.gray, fontFace: "Consolas", fontSize: 12 } },
      { text: c.pct, options: { color: c.color, bold: true, fontSize: 14 } }
    ], {
      x: 2.9 + i * 2.5, y: 4.68, w: 2.4, h: 0.55,
      fontFace: FONT, valign: "middle", margin: 0
    });
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 9 — DEPLOY E MONITORAMENTO
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };
  addTopBar(slide, C.yellow);
  addTitle(slide, "Deploy e Monitoramento");

  // Card Deploy
  addCard(slide, 0.4, 1.05, 4.4, 4.2, C.darkCard);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 1.05, w: 0.1, h: 4.2,
    fill: { color: C.yellow }, line: { color: C.yellow }
  });
  slide.addText("Deploy", {
    x: 0.65, y: 1.15, w: 3.9, h: 0.45,
    fontSize: 16, bold: true, fontFace: FONT, color: C.yellow, margin: 0
  });
  const deployItems = [
    "Docker + docker-compose",
    "Deploy em nuvem: Railway",
    "Restart automático em falhas",
    "Variável PORT dinâmica",
    "Healthcheck configurado",
  ];
  slide.addText(
    deployItems.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < deployItems.length - 1 } })),
    { x: 0.65, y: 1.7, w: 3.9, h: 3.3, fontSize: 14, fontFace: FONT, color: C.white, paraSpaceAfter: 8, margin: 0 }
  );

  // Card Monitoramento
  addCard(slide, 5.1, 1.05, 4.5, 4.2, C.darkCard);
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 5.1, y: 1.05, w: 0.1, h: 4.2,
    fill: { color: C.blue }, line: { color: C.blue }
  });
  slide.addText("Monitoramento", {
    x: 5.35, y: 1.15, w: 4.1, h: 0.45,
    fontSize: 16, bold: true, fontFace: FONT, color: C.blue, margin: 0
  });
  const monItems = [
    "Logs estruturados por requisição",
    "Tempo de resposta em ms",
    "CPU e memória em tempo real",
    "Integração com Prometheus",
    "Dashboard Grafana incluído",
  ];
  slide.addText(
    monItems.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < monItems.length - 1 } })),
    { x: 5.35, y: 1.7, w: 4.1, h: 3.3, fontSize: 14, fontFace: FONT, color: C.white, paraSpaceAfter: 8, margin: 0 }
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 10 — ENCERRAMENTO
// ═══════════════════════════════════════════════════════════════════════════
{
  const slide = pres.addSlide();
  slide.background = { color: C.bg };

  // Barras decorativas
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.2, h: 5.625,
    fill: { color: C.red }, line: { color: C.red }
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 9.8, y: 0, w: 0.2, h: 5.625,
    fill: { color: C.blue }, line: { color: C.blue }
  });

  slide.addText("Resumo do Projeto", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 30, bold: true, fontFace: FONT,
    color: C.white, align: "center"
  });

  const checks = [
    { text: "Coleta e pré-processamento",  detail: "yfinance, 1.738 registros (2018–2024)" },
    { text: "Modelo LSTM",                 detail: "MAPE de 3.62% — erro inferior a 4%" },
    { text: "API RESTful",                 detail: "FastAPI com 5 endpoints + Swagger UI" },
    { text: "Testes automatizados",        detail: "66 testes, 97% de cobertura de código" },
    { text: "Docker",                      detail: "Containerização completa + docker-compose" },
    { text: "Deploy em produção",          detail: "Railway — API disponível online" },
  ];

  checks.forEach((c, i) => {
    const y = 1.15 + i * 0.67;
    slide.addText("✅", {
      x: 0.5, y, w: 0.5, h: 0.5,
      fontSize: 18, align: "center", valign: "middle", margin: 0
    });
    slide.addText([
      { text: c.text + "  ", options: { bold: true, color: C.white } },
      { text: "— " + c.detail, options: { color: C.gray } }
    ], {
      x: 1.1, y, w: 8.4, h: 0.55,
      fontSize: 14, fontFace: FONT, valign: "middle", margin: 0
    });
  });

  // Links
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.4, y: 5.05, w: 9.2, h: 0.48,
    fill: { color: C.darkCard }, line: { color: C.blue, transparency: 50 }
  });
  slide.addText([
    { text: "GitHub: ", options: { color: C.gray } },
    { text: "github.com/rafaelbuzato/TechChallengeV4", options: { color: C.blue } },
    { text: "   |   API: ", options: { color: C.gray } },
    { text: "railway.app (produção)", options: { color: C.green } },
  ], {
    x: 0.5, y: 5.08, w: 9.0, h: 0.42,
    fontSize: 12, fontFace: FONT, align: "center", valign: "middle", margin: 0
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SALVAR
// ═══════════════════════════════════════════════════════════════════════════
const OUTPUT = "C:/Users/rafae/OneDrive/Área de Trabalho/POSTECH/TechChallengeV4/Apresentacao_TechChallenge_Fase4.pptx";

pres.writeFile({ fileName: OUTPUT })
  .then(() => console.log("✅ Apresentação salva em:\n" + OUTPUT))
  .catch(e => { console.error("❌ Erro:", e); process.exit(1); });
