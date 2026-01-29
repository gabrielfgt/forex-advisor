# Revisao Tecnica - Forex Advisor

## Sumario Executivo

O **Forex Advisor** e um sistema de analise do par cambial BRL/USD que combina analise tecnica classica, ML para explicabilidade de features e GenAI (Google Gemini) para geracao de insights contextualizados. O sistema **nao faz recomendacoes de investimento** — apenas informa e contextualiza o cenario atual do mercado.

---

## 1. Arquitetura Geral do Sistema

### 1.1 Visao do Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Yahoo Finance   │────▶│  Indicadores Tecnicos │────▶│  Classificacao      │
│  (yfinance)      │     │  (pandas/numpy)       │     │  Heuristica         │
│  5 anos OHLC     │     │  16 indicadores       │     │  (rule-based)       │
└─────────────────┘     └──────────────────────┘     └────────┬────────────┘
                                                              │
┌─────────────────┐     ┌──────────────────────┐              │
│  Google Gemini   │────▶│  Coleta de Noticias  │              │
│  (LangChain)     │     │  (3-5 noticias)      │              │
│  temp=0.5        │     │  JSON estruturado     │              │
└─────────────────┘     └──────────┬───────────┘              │
                                   │                          │
                                   ▼                          ▼
                        ┌──────────────────────────────────────┐
                        │  Agente de Insight (Gemini)          │
                        │  temp=0.7 | max_tokens=2500          │
                        │  Validacao de conteudo proibido       │
                        └──────────────────┬───────────────────┘
                                           │
                        ┌──────────────────────────────────────┐
                        │  Random Forest (explicabilidade)     │
                        │  n_estimators=100, max_depth=10      │
                        │  Feature importance ranking          │
                        └──────────────────┬───────────────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────────┐
                        │  Output: TXT formatado + console     │
                        │  outputs/classificacao_final_*.txt   │
                        └──────────────────────────────────────┘
```

### 1.2 Estrutura de Modulos

```
forex-advisor/
├── main.py                        # Orquestrador do pipeline (4 etapas)
├── src/
│   ├── data/
│   │   └── forex-scrapping.py     # Coleta OHLC via yfinance + validacao
│   ├── analysis/
│   │   └── analysis.py            # Indicadores tecnicos + classificacao + RF
│   ├── news/
│   │   └── news-scrapping.py      # Coleta de noticias via LLM
│   └── agent/
│       └── agent.py               # Geracao de insight + validacao de seguranca
├── requirements.txt
├── Dockerfile
└── .env-example
```

### 1.3 Fluxo de Execucao (main.py)

O pipeline e **sequencial e sincrono**, com 4 etapas:

1. **Etapa 1** - `fetch_forex_data(years=5)`: Coleta dados OHLC do Yahoo Finance
2. **Etapa 2** - `analyze_market(df)`: Calcula indicadores, classifica cenario, treina RF para explicabilidade
3. **Etapa 3** - `fetch_news_with_llm("BRL/USD", days=7)`: Busca noticias recentes via Gemini
4. **Etapa 4** - `generate_insight(...)`: Gera insight narrativo combinando analise tecnica + noticias

---

## 2. Machine Learning — Detalhamento

### 2.1 Modelo Utilizado: Random Forest Classifier

**Localizacao**: `src/analysis/analysis.py`, funcao `get_feature_importance()` (linha 217)

**Objetivo**: O Random Forest **nao** e usado para prever o mercado. Ele e usado exclusivamente para **explicabilidade** — calcular a importancia relativa de cada feature/indicador na classificacao heuristica.

**Configuracao**:
```python
rf = RandomForestClassifier(
    n_estimators=100,     # 100 arvores de decisao
    random_state=42,      # Reprodutibilidade
    max_depth=10          # Limita profundidade para evitar overfitting
)
```

**Pre-processamento**:
- `StandardScaler` para normalizacao das features (media 0, desvio padrao 1)
- Remocao de linhas com NaN (`dropna()`)
- Minimo de 50 amostras para treinar

**Features utilizadas no RF** (9 features):

| Feature | Descricao |
|---------|-----------|
| `SMA_20` | Media movel simples de 20 dias |
| `SMA_50` | Media movel simples de 50 dias |
| `RSI` | Indice de Forca Relativa (14 dias) |
| `BB_Width` | Largura das Bandas de Bollinger |
| `BB_Position` | Posicao do preco dentro das bandas (0-1) |
| `Volatility` | Volatilidade historica anualizada |
| `MACD` | Linha MACD |
| `MACD_Histogram` | Histograma MACD |
| `Returns` | Retornos diarios percentuais |

### 2.2 Geracao do Target (Variavel Alvo)

O target do Random Forest e gerado **sinteticamente** pela propria heuristica. Para cada ponto do historico, o sistema aplica a funcao `classify_heuristic()` no slice cumulativo dos dados ate aquele ponto:

```python
for idx in range(len(df_clean)):
    row_df = df_clean.iloc[:idx+1]
    classification = classify_heuristic(row_df)['classification']
    targets.append(classification)
```

**Implicacao critica**: O RF esta aprendendo a replicar a heuristica, nao a prever o mercado. Isso e intencional — o objetivo e extrair quais features a heuristica mais considera, nao fazer predicao.

### 2.3 Saida do RF

O output e um dicionario ordenado de `feature_name → importance_score`:

```python
feature_importance = dict(zip(available_features, rf.feature_importances_))
# Exemplo: {'RSI': 0.23, 'SMA_20': 0.19, 'Volatility': 0.16, ...}
```

Os top 5 features sao exibidos no output final.

### 2.4 Limitacoes do Uso de ML

- **Nao ha split treino/teste**: O RF e treinado em todos os dados disponiveis. Isso e aceitavel porque o objetivo nao e generalizacao preditiva, mas sim explicabilidade interna
- **Sem validacao cruzada**: Nao ha cross-validation, pela mesma razao acima
- **Target circular**: A variavel alvo vem da propria heuristica, entao o RF mede importancia relativa *da heuristica*, nao do mercado real
- **Retreinamento a cada execucao**: O modelo e treinado do zero em cada run, sem persistencia

---

## 3. Heuristicas e Indicadores Financeiros

### 3.1 Indicadores Calculados (16 indicadores)

#### Medias Moveis Simples (SMA)

```python
def calculate_sma(df, period):
    return df['Close'].rolling(window=period).mean()
```

- **SMA_20**: Tendencia de curto prazo (20 dias)
- **SMA_50**: Tendencia de medio prazo (50 dias)
- **SMA_200**: Tendencia de longo prazo (200 dias) — calculada mas **nao usada na classificacao**

#### RSI (Relative Strength Index)

```python
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

- Periodo: 14 dias (padrao da industria)
- Escala: 0 a 100
- Interpretacao: >70 sobrecomprado, <30 sobrevendido
- Na heuristica: 50-70 contribui para Alta, 30-50 contribui para Baixa

#### Bandas de Bollinger

```python
sma = Close.rolling(20).mean()
std = Close.rolling(20).std()
upper = sma + (std * 2)     # Banda superior
lower = sma - (std * 2)     # Banda inferior
```

- **BB_Width** = Upper - Lower (largura das bandas, mede volatilidade)
- **BB_Position** = (Close - Lower) / (Upper - Lower) (posicao normalizada 0-1)
- Na heuristica: largura > percentil 75 ou posicao extrema (<0.2 ou >0.8) contribui para Alta Volatilidade

#### Volatilidade Historica (Anualizada)

```python
returns = df['Close'].pct_change()
volatility = returns.rolling(20).std() * np.sqrt(252) * 100
```

- Janela: 20 dias
- Fator de anualizacao: sqrt(252) (dias uteis no ano)
- Resultado em percentual
- Na heuristica: percentil > 75 contribui para Alta Volatilidade (peso 2x)

#### MACD (Moving Average Convergence Divergence)

```python
ema_fast = Close.ewm(span=12, adjust=False).mean()
ema_slow = Close.ewm(span=26, adjust=False).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=9, adjust=False).mean()
histogram = macd_line - signal_line
```

- Parametros classicos: 12/26/9
- MACD Line: diferenca entre EMAs rapida e lenta
- Signal Line: EMA de 9 periodos da linha MACD
- Histograma: diferenca entre MACD e Signal
- **Nota**: MACD e calculado mas **nao e usado diretamente na classificacao heuristica** — apenas no RF para explicabilidade

#### Suporte e Resistencia

```python
support = df['Low'].rolling(window=20, center=True).min()
resistance = df['High'].rolling(window=20, center=True).max()
```

- Rolling window centrado de 20 dias
- Calculados mas **nao usados na classificacao** — presentes no DataFrame para enriquecimento

#### Retornos e Tendencias

- **Returns**: `Close.pct_change()` — retorno diario percentual
- **Trend_20**: `SMA_20.diff()` — direcao da SMA de 20 dias
- **Trend_50**: `SMA_50.diff()` — direcao da SMA de 50 dias

### 3.2 Sistema de Classificacao Heuristica

**Localizacao**: `src/analysis/analysis.py`, funcao `classify_heuristic()` (linha 112)

O sistema classifica o cenario em **4 categorias** usando um sistema de pontuacao (scoring):

#### Tendencia de Alta (score maximo: 4)

| Regra | Pontos |
|-------|--------|
| Preco > SMA_20 | +1 |
| Preco > SMA_50 | +1 |
| RSI entre 50 e 70 | +1 |
| Trend_20 > 0 (SMA subindo) | +1 |

#### Tendencia de Baixa (score maximo: 4)

| Regra | Pontos |
|-------|--------|
| Preco < SMA_20 | +1 |
| Preco < SMA_50 | +1 |
| RSI entre 30 e 50 | +1 |
| Trend_20 < 0 (SMA descendo) | +1 |

#### Alta Volatilidade (score maximo: 4)

| Regra | Pontos |
|-------|--------|
| Volatilidade > percentil 75 | **+2** (peso duplo) |
| BB_Width > percentil 75 | +1 |
| BB_Position < 0.2 ou > 0.8 | +1 |

#### Neutro
- Quando **nenhuma** categoria tem score > 0
- Confianca fixa: 50%

#### Calculo da Confianca

```python
confidence = min(max_score / 4.0, 1.0)  # Normalizado entre 0% e 100%
```

A confianca e o score maximo dividido por 4 (score maximo teorico), limitado a 100%.

### 3.3 Pontos Relevantes da Heuristica

- **Mutuamente nao-exclusivos**: Um cenario pode pontuar em mais de uma categoria — vence a maior pontuacao
- **Desempate**: Python `max()` retorna o primeiro elemento em caso de empate (favorece Alta sobre Baixa)
- **Peso diferenciado**: Alta Volatilidade recebe peso 2x na regra de percentil de volatilidade
- **RSI com faixas gap**: RSI < 30 e RSI > 70 **nao** contribuem para nenhuma classificacao diretamente (zonas extremas ignoradas na heuristica)

---

## 4. GenAI — Integracao com LLM

### 4.1 Provider e Modelos

| Uso | Modelo Primario | Fallback 1 | Fallback 2 |
|-----|----------------|------------|------------|
| Coleta de noticias | `gemini-2.5-flash` | `gemini-1.5-flash` | Template hardcoded |
| Geracao de insight | `gemini-2.5-flash` | `gemini-1.5-flash` | Template hardcoded |

**Framework**: LangChain 0.3+ (`langchain-google-genai`)

### 4.2 Chain de Noticias (`news-scrapping.py`)

**Tipo de chain**: `ChatPromptTemplate | ChatGoogleGenerativeAI` (LCEL — LangChain Expression Language)

**Configuracao do LLM**:
- `temperature=0.5` (mais deterministico, adequado para busca de fatos)
- `max_output_tokens=800`

**Prompt**: Solicita 3-5 noticias em formato JSON com campos `title`, `date`, `snippet`

**Parsing da resposta**:
1. Tenta extrair JSON via regex: `re.search(r'\{.*\}', content, re.DOTALL)`
2. Se falhar JSON, faz parsing linha-a-linha (`_parse_llm_response`)
3. Se tudo falhar, usa fallback com noticia generica

**Ponto de atencao**: O LLM nao acessa a internet em tempo real. Ele gera noticias baseadas no seu **knowledge cutoff**. Isso significa que as "noticias" podem nao ser atuais.

### 4.3 Chain de Insight (`agent.py`)

**Tipo de chain**: `ChatPromptTemplate | ChatGoogleGenerativeAI` (LCEL)

**Configuracao do LLM**:
- `temperature=0.7` (mais criativo, adequado para narrativa)
- `max_output_tokens=2500`

**System message**: `"Voce e um analista financeiro que fornece informacoes contextuais sobre cambio, sem fazer recomendacoes de investimento."`

**Prompt estruturado** (`_build_prompt`):
Recebe e formata:
- Classificacao tecnica (tipo + confianca + explicacao)
- Indicadores principais (preco, SMAs, RSI, volatilidade)
- Lista de noticias formatada
- Instrucoes explicitas de formato (3-4 frases, informativo, sem recomendacoes)

### 4.4 O Que NAO E Usado

O projeto **nao utiliza**:
- **Agents** (LangChain Agent framework com tools/actions)
- **RAG** (Retrieval-Augmented Generation)
- **Vector stores** ou embeddings
- **Memory/ConversationBufferMemory**
- **Tool calling** ou function calling
- **ReAct loop**

A arquitetura e puramente **sequential chain** — dois LLM calls independentes (noticias e insight), sem loop de raciocinio.

### 4.5 Sistema de Validacao de Output (Safety Guard)

**Localizacao**: `src/agent/agent.py`, funcao `_validate_insight()` (linha 270)

#### Lista de palavras proibidas (FORBIDDEN_WORDS):

```python
FORBIDDEN_WORDS = [
    'compre agora', 'venda agora', 'compre', 'venda', 'invista',
    'nao invista', 'e o momento de comprar', 'e o momento de vender',
    'deve comprar', 'deve vender', 'recomendo comprar', 'recomendo vender',
    'sugiro comprar', 'sugiro vender', 'buy now', 'sell now',
    'you should buy', 'you should sell', 'invest now'
]
```

#### Padroes regex de recomendacao:

```python
recommendation_patterns = [
    r'voce (deve|precisa|pode) (comprar|vender|investir)',
    r'(e|seria) (recomendavel|aconselhavel) (comprar|vender|investir)',
    r'(sugiro|recomendo|aconselho) (que voce )?(comprar|vender|investir)'
]
```

#### Pipeline de validacao:

1. **Checagem de tamanho minimo**: insight < 50 caracteres → fallback
2. **Checagem de truncamento**: sem pontuacao final + < 100 chars → adiciona ponto ou fallback
3. **Substituicao de palavras proibidas**: palavra proibida → `"[informacao removida]"`
4. **Substituicao de padroes regex**: padrao de recomendacao → `"[informacao contextual]"`
5. **Checagem pos-substituicao**: resultado < 20 chars → fallback

### 4.6 Mecanismo de Fallback Completo

```
Gemini 2.5 Flash
    ↓ (falha)
Gemini 1.5 Flash
    ↓ (falha)
Gemini 1.5 Flash sem system message
    ↓ (falha)
Template hardcoded (_generate_fallback)
```

O template de fallback gera uma frase simples concatenando classificacao + explicacao + preco + contexto generico de noticias.

---

## 5. Pipeline de Dados

### 5.1 Coleta (forex-scrapping.py)

- **Fonte**: Yahoo Finance via `yfinance`
- **Ticker**: `BRL=X` (taxa de cambio BRL/USD)
- **Periodo padrao**: 5 anos
- **Colunas**: Date, Open, High, Low, Close, Volume

### 5.2 Validacao de Dados (`validate_data`)

1. **NaN removal**: `dropna(subset=['Open', 'High', 'Low', 'Close'])`
2. **Consistencia OHLC**: verifica que High >= max(Open, Close), Low <= min(Open, Close), High >= Low
3. **Deteccao de outliers** (metodo IQR):
   - Q1 e Q3 calculados por coluna
   - IQR = Q3 - Q1
   - Limites: Q1 - 3*IQR ate Q3 + 3*IQR
   - Outliers sao **logados mas nao removidos** (preserva integridade dos dados)
4. **Ordenacao** por data + reset do indice

### 5.3 Fluxo de Transformacao

```
Dados brutos (OHLC) → 6 colunas
    ↓ calculate_all_indicators()
DataFrame enriquecido → ~22 colunas
    ↓ classify_heuristic()
Classificacao (dict) → classification, confidence, explanation, scores
    ↓ get_feature_importance()
Feature importance (dict) → feature_name → score
```

---

## 6. Stack Tecnologico

| Camada | Tecnologia | Versao |
|--------|-----------|--------|
| Dados de mercado | yfinance | >= 1.0 |
| Processamento | pandas, numpy | >= 2.0, >= 1.24 |
| ML | scikit-learn | >= 1.3 |
| LLM Framework | LangChain | >= 0.3 |
| LLM Provider | Google Gemini (langchain-google-genai) | >= 2.0 |
| HTTP | requests | >= 2.31 |
| Scraping (preparado) | beautifulsoup4 | >= 4.12 |
| Config | python-dotenv | >= 1.0 |
| Runtime | Python | 3.11+ |
| Container | Docker (python:3.11-slim) | — |

---

## 7. Consideracoes de Producao (do README)

O README menciona as seguintes estrategias para escalar:

- **Cache de insights**: TTL de 1 hora, chave = `hash(datetime + par_moedas)`
- **Fila de mensagens**: RabbitMQ ou Kafka para processamento assincrono
- **Workers dedicados**: data-collector (15 min), technical-analyzer (real-time), news-scraper (1h), insight-generator (on-demand)
- **Vector DB**: PostgreSQL + PGVector para embeddings de noticias
- **Cache**: Redis em cluster mode
- **API**: FastAPI com rate limiting
- **Monitoramento**: Prometheus + Grafana

---

## 8. Pontos Fortes do Projeto

1. **Separacao clara de responsabilidades**: cada modulo tem uma unica funcao
2. **Explicabilidade sobre predicao**: prioriza entender o "porque" sobre o "o que vai acontecer"
3. **Safety guards robustos**: validacao multi-camada contra recomendacoes de investimento
4. **Graceful degradation**: fallbacks em todos os niveis (LLM, dados, output)
5. **Reprodutibilidade**: `random_state=42` no RF, parametros configurados via .env
6. **Containerizacao**: Docker-ready para deploy
7. **Heuristica interpretavel**: regras claras e auditaveis de classificacao

## 9. Pontos de Melhoria / Fragilidades

1. **Noticias nao sao reais em tempo real**: O LLM gera noticias a partir do seu knowledge cutoff, nao scraping real
2. **Sem persistencia de modelo**: RF e retreinado do zero a cada execucao
3. **Sem testes automatizados**: nenhum arquivo de testes unitarios ou integracao
4. **Target circular do RF**: treinado para replicar a heuristica, nao para aprender padroes de mercado reais
5. **Sem API REST**: apenas execucao via CLI
6. **Importacao via importlib**: modulos com hifen no nome exigem `importlib.util`, fragilidade de manutencao
7. **Sem monitoramento de drift**: nenhuma metrica sobre mudanca no comportamento do modelo ao longo do tempo
8. **SMA_200 e Suporte/Resistencia calculados mas nao usados na classificacao**

---

## 10. Perguntas e Respostas para Entrevista

### 10.1 Perguntas de Tech Lead

---

**P1: Por que voce escolheu uma heuristica baseada em regras ao inves de um modelo de ML puro para classificacao?**

**R:** A escolha foi deliberada. No dominio financeiro, **interpretabilidade e auditabilidade** sao mais valiosas que acuracia preditiva marginal. Um modelo de ML puro (como uma rede neural) seria uma caixa preta — se a classificacao indicar "Tendencia de Alta", ninguem saberia explicar por que. Com regras explicitas (preco > SMA_20, RSI entre 50-70, etc.), qualquer analista consegue auditar e entender a decisao. Alem disso, o projeto nao pretende prever o mercado — apenas categorizar o cenario atual para contextualizar um insight. Para esse objetivo, regras simples e transparentes sao mais adequadas.

---

**P2: Qual o papel real do Random Forest no sistema? Ele esta fazendo predicao?**

**R:** Nao. O Random Forest nao faz predicao nenhuma no pipeline de producao. Ele e treinado como modelo auxiliar exclusivamente para **explicabilidade de features**. O target dele e gerado pela propria heuristica — entao ele esta aprendendo "quais indicadores mais influenciam as regras que eu mesmo defini". O output dele e um ranking de `feature_importances_`, que e exibido ao usuario para entender quais indicadores estao sendo mais relevantes. E um uso consciente de ML como ferramenta de interpretacao, nao de predicao.

---

**P3: Voce tem consciencia de que o target do RF e circular? O modelo esta aprendendo a replicar suas proprias regras.**

**R:** Sim, e proposital. O RF nao pretende generalizar para dados nao vistos — nao ha split treino/teste e nao ha validacao cruzada porque o objetivo nao e avaliacao de performance preditiva. O valor esta no `feature_importances_`: ele quantifica a contribuicao relativa de cada feature para as decisoes da heuristica. Em producao, isso serve para **comunicacao com stakeholders** — "RSI contribuiu 23% para a classificacao atual" e mais informativo que "ativou a regra 3". Se fosse preditivo, a circularidade seria um problema grave; como e explicativo, e uma ferramenta valida.

---

**P4: Como voce garante que o sistema nao faz recomendacoes de investimento?**

**R:** Ha tres camadas de protecao:
1. **Prompt engineering**: O system message e o prompt humano instruem explicitamente o LLM a nunca fazer recomendacoes, com frases como "NUNCA faca recomendacoes de compra/venda"
2. **Validacao pos-geracao**: A funcao `_validate_insight()` verifica o output contra uma lista de 15+ palavras proibidas (em PT e EN) e 3 padroes regex de recomendacao. Qualquer match e substituido por `[informacao removida]`
3. **Fallback seguro**: Se a validacao falhar ou o insight for truncado/curto demais, um template hardcoded e usado — e esse template nunca contem recomendacoes

---

**P5: Se voce fosse escalar este sistema para producao real, quais seriam as primeiras 3 mudancas?**

**R:** As tres primeiras acoes seriam:
1. **Substituir a coleta de noticias por scraping real**: Atualmente o LLM gera noticias a partir do knowledge cutoff, que podem estar desatualizadas. Em producao, usaria RSS feeds de Reuters/Bloomberg, APIs de noticias (NewsAPI, GDELT), ou scraping com BeautifulSoup — e ai sim passaria as noticias reais para o LLM contextualizar
2. **Adicionar uma API REST** (FastAPI): O sistema atual e CLI-only. Uma API permitiria integracao com dashboards, alertas e outros servicos
3. **Implementar cache e persistencia**: Usar Redis para cachear classificacoes e insights com TTL de 1 hora, e um banco (PostgreSQL) para historico de analises. Evitaria retreinar o RF e rechamar o LLM a cada request

---

**P6: Por que nao ha testes no projeto?**

**R:** Este e um ponto de melhoria reconhecido. Em uma evolucao, eu adicionaria:
- **Testes unitarios** para cada funcao de calculo de indicadores (validar formulas contra valores conhecidos)
- **Testes de integracao** para o pipeline completo (mock do yfinance + mock do Gemini)
- **Testes da validacao de seguranca**: garantir que todos os FORBIDDEN_WORDS sao efetivamente substituidos
- **Testes de contrato do LLM**: verificar que o prompt gera outputs no formato esperado, usando prompts de referencia

---

### 10.2 Perguntas de Arquiteto

---

**P7: Por que LangChain ao inves de chamar a API do Gemini diretamente?**

**R:** LangChain foi escolhido por tres razoes:
1. **Abstracacao de provider**: Se amanha quisermos trocar Gemini por OpenAI, Anthropic ou Ollama, basta mudar o construtor do LLM. O `ChatPromptTemplate` e o operador `|` (pipe) sao provider-agnostic
2. **LCEL (LangChain Expression Language)**: O padrao `prompt | llm` facilita composicao de chains. Hoje e simples, mas se precisar adicionar output parsers, retrievers ou routers, a estrutura ja esta pronta
3. **Ecossistema**: LangChain oferece integracao nativa com tracing (LangSmith), streaming, callbacks e retry — uteis em producao

Dito isso, para o uso atual (dois calls sequenciais simples), a API direta do Gemini seria igualmente funcional e com menos overhead de dependencias.

---

**P8: Voce usa Agents do LangChain? Se nao, por que?**

**R:** Nao. O sistema nao usa Agents (ReAct loop, tool calling, function schemas). O fluxo e **sequencial e deterministico**: dados → indicadores → classificacao → noticias → insight. Nao ha decisao dinamica do LLM sobre qual ferramenta usar ou se deve iterar. Um Agent seria over-engineering para este caso: adicionaria latencia, custo e imprevisibilidade sem beneficio claro. Se no futuro o sistema precisasse decidir *quais* indicadores calcular, ou *quais* fontes consultar baseado no contexto, ai sim um Agent faria sentido.

---

**P9: Como voce trataria latencia do LLM em producao?**

**R:** Varias estrategias:
1. **Cache**: Insights com TTL de 1 hora. Key = hash(classificacao + data). A classificacao tecnica muda lentamente, entao muitos requests podem usar cache
2. **Pre-geracao**: Worker que gera insights para as 4 classificacoes possiveis periodicamente. Requests servem do cache
3. **Streaming**: Usar `llm.stream()` do LangChain para enviar tokens incrementalmente ao cliente
4. **Modelo mais leve**: `gemini-1.5-flash` ao inves de `gemini-2.5-flash` para casos onde latencia e mais importante que qualidade
5. **Processamento assincrono**: Fila (RabbitMQ/Kafka) onde o request entra e o insight e entregue via webhook ou polling

---

**P10: Qual o risco de alucinacao do LLM neste sistema?**

**R:** Ha dois pontos criticos de alucinacao:
1. **Noticias fabricadas**: O LLM nao tem acesso a internet. Ele gera "noticias" a partir do seu knowledge cutoff. Podem ser plausíveis mas desatualizadas ou inventadas. **Mitigacao**: Em producao, substituir por noticias reais via API/scraping
2. **Insight inconsistente com dados**: O LLM pode gerar uma narrativa que contradiz os indicadores (ex: dizer "mercado calmo" quando a volatilidade esta no percentil 95). **Mitigacao**: A validacao atual checa palavras proibidas, mas nao verifica consistencia semantica. Uma melhoria seria adicionar checagens programaticas (ex: se volatilidade > 75 percentil, o insight deve mencionar volatilidade)

---

**P11: Se voce precisasse adicionar RAG ao projeto, como faria?**

**R:** O ponto natural de integracao seria na etapa de noticias:
1. **Ingestao**: Coletar noticias reais via RSS/APIs e gerar embeddings com um modelo de embedding (ex: `models/text-embedding-004` do Google ou `text-embedding-3-small` da OpenAI)
2. **Armazenamento**: Vector store — PostgreSQL + PGVector (como mencionado no README) ou ChromaDB para prototipo rapido
3. **Retrieval**: Ao gerar o insight, buscar top-k noticias semanticamente similares ao contexto atual (classificacao + indicadores)
4. **Augmented Generation**: Passar as noticias reais como contexto no prompt do LLM, substituindo a geracao baseada em knowledge cutoff

Isso resolveria o problema de noticias fabricadas e daria ao insight informacao factual atualizada.

---

**P12: A arquitetura modular permite trocar o par de moedas facilmente?**

**R:** Parcialmente. O ticker `BRL=X` esta hardcoded no `fetch_forex_data()` e em diversas strings de display no `main.py`. Para suportar multiplos pares, seria necessario:
1. Parametrizar o ticker em `fetch_forex_data(ticker="BRL=X")`
2. Propagar o par de moedas como parametro em todo o pipeline
3. Ajustar os prompts do LLM para refletir o par correto (ja parcialmente feito em `news-scrapping.py` que recebe `currency_pair` como parametro)
4. Validar que os indicadores fazem sentido para o novo par (os indicadores tecnicos sao universais, mas os thresholds da heuristica podem precisar de ajuste)

---

### 10.3 Perguntas de Staff de GenAI

---

**P13: Como voce aborda prompt engineering neste projeto? Qual a estrategia de design dos prompts?**

**R:** Ha dois prompts distintos com estrategias diferentes:

**Prompt de noticias** (temperatura 0.5):
- Instrucao de role: "assistente especializado em analise financeira"
- Output estruturado: solicita JSON com schema definido
- Fallback parsing: se o JSON falhar, parsing linha-a-linha
- Quantidade delimitada: "3-5 noticias"
- Escape clause: "Se nao encontrar noticias especificas, mencione eventos gerais"

**Prompt de insight** (temperatura 0.7):
- **Dual-role instruction**: system message define o papel, human prompt da o contexto
- **Constraint-based prompting**: instrucoes explicitas do que NAO fazer (nunca recomendar compra/venda)
- **Structured context injection**: indicadores formatados como lista bullet-point
- **Output specification**: "3-4 frases", "profissional mas acessivel"
- **Reinforcement**: lembrete final "Voce esta apenas informando, nao recomendando"

A estrategia geral e **constraint-first**: define o que o LLM nao pode fazer antes de definir o que deve fazer. Isso e eficaz para compliance.

---

**P14: A temperatura 0.7 nao e alta demais para um sistema financeiro?**

**R:** E uma decisao de trade-off:
- **Temperatura 0.7** e usada no insight porque queremos **variabilidade narrativa** — se sempre gerasse o mesmo texto, nao teria valor. O insight e informativo, nao decisorio
- **Temperatura 0.5** e usada nas noticias porque queremos **fidelidade factual** (na medida do possivel sem acesso a internet)
- Uma alternativa seria usar **temperatura 0.3-0.4** para insights tambem, com `top_p` para controlar diversidade. Mas como ha validacao pos-geracao, o risco de output inadequado e mitigado

---

**P15: Como voce avalia a qualidade dos outputs do LLM? Ha alguma metrica?**

**R:** Atualmente nao ha metricas formais. As checagens sao binarias:
- Output nao-vazio? (>50 chars)
- Output nao-truncado? (termina com pontuacao)
- Output sem palavras proibidas?

Para evolucao, eu implementaria:
1. **Consistencia factual**: Checagem programatica de que o insight menciona a classificacao correta e indicadores compativeis
2. **LLM-as-judge**: Usar um segundo LLM call (barato, como Gemini Flash) para avaliar se o insight e informativo, coerente e livre de recomendacoes
3. **Human evaluation loop**: Logging dos insights gerados para revisao periodica por analistas
4. **Metricas de engagement**: Em producao com API, medir quais insights sao mais lidos/uteis

---

**P16: O sistema tem alguma forma de memoria ou contexto entre execucoes?**

**R:** Nao. Cada execucao e completamente **stateless**:
- O RF e treinado do zero
- O LLM nao tem memoria de conversas anteriores
- Nao ha banco de dados ou cache persistente
- O output anterior nao influencia o proximo

Para adicionar memoria, as opcoes seriam:
1. **Cache Redis**: Ultimas N classificacoes para detectar mudanca de cenario ("mudou de Alta para Baixa nas ultimas 24h")
2. **Historico de insights**: Banco SQL com insights anteriores para evitar repeticao e mostrar evolucao
3. **ConversationBufferMemory do LangChain**: Se houvesse interacao conversacional (chatbot), mas nao e o caso atual
4. **Embedding similarity**: Comparar insight atual com anteriores para garantir diversidade

---

**P17: Voce considerou usar function calling / tool use com o Gemini?**

**R:** Considerei, mas nao implementei. Function calling seria util se o LLM precisasse:
- Consultar APIs externas de forma dinamica (ex: "buscar a taxa Selic atual")
- Escolher qual indicador calcular baseado no contexto
- Executar queries em banco de dados

No design atual, **todas as informacoes necessarias ja sao calculadas antes do LLM call** e passadas como contexto no prompt. O LLM apenas consome dados pre-processados e gera texto. Function calling adicionaria complexidade e latencia sem beneficio no fluxo atual. Se o sistema evoluisse para um chatbot interativo onde o usuario pode perguntar "qual o RSI historico de janeiro?", ai sim function calling seria essencial.

---

**P18: Como voce garantiria que o prompt nao e vulneravel a injection?**

**R:** No cenario atual, o risco de prompt injection e baixo porque:
- O input do usuario **nao existe** — nao ha interface de input humano. Todos os dados vem de APIs (yfinance) ou sao calculados internamente
- As noticias vem do proprio LLM (entao seriam "self-injection", que e diferente)

Se houvesse input do usuario (ex: chatbot), as mitigacoes seriam:
1. **Input sanitization**: Limitar caracteres especiais e tamanho
2. **Prompt isolation**: Separar system prompt do user input com delimitadores fortes
3. **Output validation**: Ja existe (FORBIDDEN_WORDS + regex patterns)
4. **Guardrails**: Usar frameworks como NeMo Guardrails ou Guardrails AI para validacao de input/output
5. **Least privilege**: O LLM nao tem acesso a tools ou funcoes criticas

---

**P19: Se voce pudesse redesenhar o sistema com arquitetura agentica moderna, como ficaria?**

**R:** Um redesign agentico usaria:

```
┌─────────────────────────────────────────────────┐
│             Orchestrator Agent                   │
│  (decide quais sub-agentes chamar)              │
└───────────┬─────────────┬───────────────────────┘
            │             │
    ┌───────▼──────┐ ┌────▼──────────────┐
    │ Data Agent   │ │ News Agent         │
    │ Tools:       │ │ Tools:             │
    │ - yfinance   │ │ - NewsAPI          │
    │ - indicators │ │ - RSS scraper      │
    │ - validate   │ │ - Google Search    │
    └───────┬──────┘ │ - Semantic search  │
            │        └────┬──────────────┘
            │             │
    ┌───────▼─────────────▼──────────────┐
    │         Analysis Agent              │
    │  Tools:                             │
    │  - classify(indicators)             │
    │  - feature_importance(rf)           │
    │  - compare_with_history(db)         │
    └───────────────┬────────────────────┘
                    │
    ┌───────────────▼────────────────────┐
    │         Insight Agent               │
    │  - Gera narrativa                   │
    │  - Auto-valida contra guidelines    │
    │  - Enriquece com contexto historico │
    └───────────────┬────────────────────┘
                    │
    ┌───────────────▼────────────────────┐
    │     Compliance Agent                │
    │  - Valida ausencia de recomendacoes│
    │  - Checa consistencia com dados    │
    │  - Aprova ou rejeita output        │
    └────────────────────────────────────┘
```

Beneficios: cada agente e especializado, pode ser testado isoladamente, e o orchestrator pode decidir dinamicamente se precisa de mais contexto.

---

**P20: Qual a diferenca entre usar LCEL (pipe operator) e criar um Agent com tools no LangChain?**

**R:**

| Aspecto | LCEL (pipe) | Agent com Tools |
|---------|------------|-----------------|
| Fluxo | Deterministico, linear | Dinamico, iterativo (ReAct loop) |
| Decisao | Definida pelo desenvolvedor | Definida pelo LLM |
| Latencia | 1 LLM call | N LLM calls (media 2-5) |
| Custo | Previsivel | Variavel |
| Debugging | Facil (fluxo fixo) | Complexo (depende do raciocinio do LLM) |
| Caso de uso | Pipeline fixo com dados pre-processados | Tarefas abertas que exigem raciocinio e acesso a ferramentas |

O Forex Advisor usa LCEL porque o fluxo e fixo e todos os dados necessarios sao coletados antes do LLM call. Um Agent seria mais adequado se o LLM precisasse decidir "preciso de mais dados" ou "vou consultar outra fonte".

---

**P21: Como voce monitoraria este sistema de GenAI em producao?**

**R:** Implementaria monitoramento em 4 dimensoes:

1. **Operacional**: Latencia dos LLM calls, taxa de erros, uso de fallbacks, custo por request (tokens in/out)
2. **Qualidade**: Score de consistencia (insight vs indicadores), taxa de intervencao da validacao (quantos insights precisaram de substituicao de palavras), comprimento medio do insight
3. **Seguranca/Compliance**: Taxa de ativacao de FORBIDDEN_WORDS, logs de todos os outputs filtrados, alertas se a taxa subir
4. **Drift**: Comparar distribuicao de classificacoes ao longo do tempo, monitorar se o RF muda significativamente o ranking de features

Ferramentas: **LangSmith** (tracing de chains), **Prometheus + Grafana** (metricas operacionais), **Logging estruturado** (ELK stack para auditoria)

---

**P22: Se o Gemini ficar fora do ar em producao, o que acontece?**

**R:** O sistema tem **graceful degradation** em multiplos niveis:

1. `gemini-2.5-flash` falha → tenta `gemini-1.5-flash`
2. `gemini-1.5-flash` falha → tenta `gemini-1.5-flash` sem system message
3. Tudo falha → usa template hardcoded (`_generate_fallback`)

O template de fallback gera um insight baseado apenas nos dados tecnicos, sem narrativa rica. E funcional mas menos informativo. A classificacao tecnica e o ranking de features **nao dependem do LLM** — sao puramente computacionais. Entao o sistema continua fornecendo valor mesmo com LLM indisponivel — apenas o insight narrativo e degradado.

---

**P23: Qual a diferenca de temperatura entre os dois LLM calls e por que?**

**R:**

| Call | Temperatura | Justificativa |
|------|-------------|---------------|
| Noticias | 0.5 | Busca de informacao factual. Menor temperatura = menor criatividade = menor risco de fabricacao |
| Insight | 0.7 | Geracao de narrativa informativa. Maior temperatura = mais variabilidade textual = insights menos repetitivos |

A diferenca reflete a natureza da tarefa: **busca de fatos** (deterministico) vs **criacao de narrativa** (criativo). Se ambos usassem 0.5, os insights seriam repetitivos. Se ambos usassem 0.7, as noticias teriam mais risco de imprecisao.

---

**P24: Como voce testaria os prompts do LLM de forma sistematica?**

**R:** Implementaria uma **evaluation suite**:

1. **Golden dataset**: 20-30 combinacoes de (classificacao, indicadores, noticias) com insights de referencia escritos por humanos
2. **Automated eval**: Para cada caso do golden dataset, gerar insight com o LLM e avaliar:
   - Presenca de palavras proibidas (deve ser 0)
   - Mencao da classificacao correta
   - Comprimento dentro do range esperado (3-4 frases)
   - Coerencia com indicadores (ex: menciona volatilidade se esta alta)
3. **LLM-as-judge**: Usar Claude ou GPT-4 para avaliar qualidade, coerencia e tom dos insights gerados
4. **Regression testing**: Rodar a suite a cada mudanca de prompt ou troca de modelo para detectar regressoes
5. **A/B testing**: Em producao, comparar versoes de prompt medindo metricas de qualidade

---

**P25: O que voce faria diferente se comecasse o projeto do zero hoje?**

**R:** Cinco mudancas principais:
1. **Noticias reais desde o inicio**: Integracao com NewsAPI ou Google News RSS, com fallback para LLM apenas se APIs falharem
2. **FastAPI desde o inicio**: Expor como API REST desde o dia 1, mesmo que simples
3. **Testes desde o dia 1**: Pelo menos testes unitarios para calculos de indicadores e validacao de insight
4. **Observabilidade nativa**: LangSmith para tracing de LLM calls, logging estruturado com correlation IDs
5. **Nomes de arquivo sem hifen**: `forex_scrapping.py` ao inves de `forex-scrapping.py`, para evitar a necessidade de `importlib.util` e permitir imports normais do Python
