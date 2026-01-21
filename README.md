# Forex Advisor

Sistema de análise de mercado de câmbio baseado em pipeline de dados, coleta automatizada de notícias, classificação utilizando Machine Learning clássico e Large Language Models (LLM) para geração de insights contextualizados sobre o par de moedas BRL/USD.

## Visão Geral

O Forex Advisor combina análise técnica quantitativa (indicadores técnicos) com dados qualitativos (notícias recentes) para gerar insights informativos que ajudam usuários a entender o cenário atual do mercado de câmbio, **sem fazer recomendações explícitas de investimento**.

### Objetivo

O objetivo do sistema é fornecer informações contextuais claras e objetivas sobre o mercado de câmbio, auxiliando usuários na tomada de decisões informadas através da análise técnica e contextualização de eventos de mercado.

## Instalação e Execução

### Pré-requisitos

- Python 3.11 ou superior
- Docker (opcional, para execução via container)
- API Key do Google Gemini (opcional - o sistema funciona com fallback quando não configurada)

### Instalação Local

1. Clone o repositório:
```bash
git clone <repository-url>
cd forex-advisor
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente (opcional):
```bash
# Criar arquivo .env
export GOOGLE_API_KEY="sua_chave_aqui"  # Para utilizar Google Gemini
export LLM_PROVIDER="gemini"  # Padrão: "gemini".
```

5. Execute o projeto:
```bash
python main.py
```

## Execução via Docker

1. Construa a imagem Docker:
```bash
docker build -t forex-advisor .
```

2. Execute o container:
```bash
# Com arquivo de variáveis de ambiente
docker run --env-file .env forex-advisor

# Ou passando variáveis diretamente
docker run -e GOOGLE_API_KEY="sua_chave" -e LLM_PROVIDER="gemini" forex-advisor
```

## Lógica do Motor de Recomendação

### Indicadores Técnicos Utilizados

O sistema calcula os seguintes indicadores técnicos:

1. **Médias Móveis (SMA)**:
   - SMA(20): Média móvel de 20 dias
   - SMA(50): Média móvel de 50 dias
   - SMA(200): Média móvel de 200 dias

2. **RSI (Relative Strength Index)**:
   - Período: 14 dias
   - Mede momentum e identifica condições de sobrecompra/sobrevenda

3. **Bandas de Bollinger**:
   - Base: SMA(20)
   - Desvios: ±2 desvios padrão
   - Identifica volatilidade e níveis de suporte/resistência dinâmicos

4. **Volatilidade Histórica**:
   - Janela: 20 dias
   - Calculada como desvio padrão dos retornos (anualizada)

5. **MACD (Moving Average Convergence Divergence)**:
   - Fast: 12 períodos
   - Slow: 26 períodos
   - Signal: 9 períodos
   - Identifica mudanças de tendência

6. **Suporte e Resistência**:
   - Calculados usando mínimos e máximos locais (janela de 20 dias)

### Heurística de Classificação

O sistema classifica o cenário atual do mercado em **4 categorias** utilizando uma heurística baseada em regras:

#### 1. Tendência de Alta

**Condições:**
- Preço atual acima da SMA(20) e SMA(50)
- RSI entre 50-70 (zona neutra-alta)
- Tendência ascendente (SMA(20) em alta)

**Pontuação**: +1 ponto para cada condição atendida

#### 2. Tendência de Baixa

**Condições:**
- Preço atual abaixo da SMA(20) e SMA(50)
- RSI entre 30-50 (zona neutra-baixa)
- Tendência descendente (SMA(20) em queda)

**Pontuação**: +1 ponto para cada condição atendida

#### 3. Alta Volatilidade

**Condições:**
- Volatilidade histórica acima do percentil 75
- Bandas de Bollinger expandidas (largura acima do percentil 75)
- Preço próximo das bandas (posição < 0.2 ou > 0.8)

**Pontuação**: +2 pontos para volatilidade alta, +1 para cada outra condição

#### 4. Neutro

**Condição:** Quando nenhuma das outras categorias se destaca claramente

### Exemplo de Classificação

```
Classificação: Tendência de Alta
Confiança: 75%
Explicação: Preço (5.1234) acima das médias móveis; RSI em 62.50 (zona neutra-alta)
```

## Explicabilidade das Features

### Método de Explicabilidade

O sistema utiliza um **Random Forest Classifier** como modelo auxiliar para calcular a importância de cada feature. O modelo é treinado utilizando as classificações heurísticas como target, permitindo identificar quais indicadores técnicos são mais influentes na classificação do mercado.

### Features Calculadas

| Feature | Descrição | Importância Típica |
|---------|-----------|-------------------|
| `RSI` | Relative Strength Index | Alta |
| `SMA_20` | Média Móvel de 20 dias | Alta |
| `SMA_50` | Média Móvel de 50 dias | Média-Alta |
| `Volatility` | Volatilidade histórica | Média |
| `BB_Width` | Largura das Bandas de Bollinger | Média |
| `BB_Position` | Posição do preço nas bandas | Média |
| `MACD` | MACD Line | Média-Baixa |
| `MACD_Histogram` | Histograma do MACD | Baixa |
| `Returns` | Retornos diários | Baixa |

### Interpretação das Contribuições

- **Importância > 0.15**: Feature altamente influente na classificação
- **Importância 0.10-0.15**: Feature moderadamente influente
- **Importância < 0.10**: Feature com menor influência

### Exemplo de Output

```
FEATURES MAIS INFLUENTES
   RSI: 0.2341
   SMA_20: 0.1892
   Volatility: 0.1567
   BB_Width: 0.1234
   SMA_50: 0.0987
```

## Pipeline de Geração de Insights

### Fluxo de Dados

```
Dados OHLC (yfinance)
    ↓
Análise Técnica (Indicadores + Classificação)
    ↓
Coleta de Notícias (LLM)
    ↓
Agente LLM (Geração de Insights)
    ↓
Output Final (Insight Contextualizado)
```

### Integração de Notícias

1. **Busca de Contexto**: O sistema utiliza Google Gemini (LLM) para buscar e resumir notícias recentes (últimos 7 dias) relevantes para o par BRL/USD
2. **Filtragem**: As notícias são filtradas por relevância e data
3. **Formatação**: As notícias são formatadas para inclusão no prompt do agente LLM

### Formato do Prompt

O prompt enviado ao LLM inclui:
- Classificação técnica atual e explicação
- Valores dos indicadores principais
- Contexto de notícias recentes
- **Instrução explícita**: "NUNCA faça recomendações de compra/venda, apenas informe e contextualize"

### Validação de Output

O sistema valida o output gerado para garantir:
- Ausência de palavras proibidas (ex: "compre agora", "venda")
- Ausência de padrões de recomendação
- Tamanho adequado (3-4 frases)
- Aplicação de correções automáticas quando necessário

### Exemplo de Insight Gerado

```
O mercado de BRL/USD apresenta uma classificação de 'Tendência de Alta' com 
confiança de 75%. O preço atual está acima das médias móveis de 20 e 50 dias, 
com RSI em 62.5, indicando momentum positivo. Notícias recentes sobre 
indicadores econômicos do Brasil podem estar influenciando a volatilidade do 
mercado. Recomenda-se monitorar continuamente os indicadores técnicos e 
eventos econômicos relevantes.
```

## Escalabilidade

Esta seção detalha como o sistema pode ser escalado para atender milhares de usuários ativos, garantindo performance, atualização e relevância dos dados.

### LLM em Produção

**Estratégia**: Cache de insights pré-gerados + geração sob demanda

**Abordagem**:
- **Cache por Classificação + Contexto**: Gerar insights em batch para combinações comuns de classificação técnica e contexto de notícias
- **TTL de Cache**: 1 hora (dados de mercado mudam rapidamente)
- **Fallback em Tempo Real**: Apenas para combinações não cacheadas ou quando cache expira
- **Chave de Cache**: `hash(datetime + moeda/par forex/ticker)`

**Benefícios:**
- Redução de custos de API do LLM
- Redução de latência
- Capacidade de atender vários usuários simultaneamente
- Disponibilidade via API

### Injeção de Contexto

**Estratégia**: RAG (Retrieval Augmented Generation) para notícias + Injeção direta para dados técnicos

#### RAG para Notícias

1. **Vector Database**: Armazenar embeddings de notícias históricas
   - **Opções**: Postgre + PGVector
   - **Embeddings**: Usar modelo como `text-embedding-ada-002` (OpenAI), `text-embedding-004` (Google) ou `sentence-transformers`

2. **Pipeline de Atualização**:
   - **Coleta Assíncrona**: Workers que coletam notícias a cada hora
   - **Geração de Embeddings**: Processar novas notícias e gerar embeddings
   - **Indexação**: Atualizar vector DB com novos embeddings

3. **Busca Semântica**:
   - **Query**: "BRL/USD exchange rate Brazil economy"
   - **Retrieval**: Top-K notícias mais relevantes (K=5-10)
   - **Injeção**: Incluir no prompt do LLM

#### Dados de Trading

- **Injeção Direta**: Dados técnicos são leves (~500 bytes) e podem ser injetados diretamente no prompt
- **Sem RAG Necessário**: Não requer busca semântica, apenas formatação

### Estratégias de Cache

#### 1. Classificação do Dia
- **Cache Key**: `classification:{date}:{currency_pair}`
- **TTL**: 1 hora (mudanças intradiárias)
- **Storage**: Redis
- **Atualização**: Recalcular apenas se dados de mercado atualizados

#### 2. Insights Gerados
- **Cache Key**: `insight:{classification_hash}:{news_hash}:{date}`
- **TTL**: 1 hora
- **Storage**: Redis
- **Invalidation**: Quando classificação ou notícias mudam significativamente

#### 3. Embeddings de Notícias
- **Cache**: Permanente até substituição por notícias mais recentes
- **Storage**: Vector DB (Pinecone/Weaviate)
- **Atualização**: Pipeline assíncrono a cada hora
- **Retention**: Manter últimos 30 dias

#### 4. Indicadores Técnicos
- **Cache Key**: `indicators:{date}:{currency_pair}`
- **TTL**: 15 minutos
- **Storage**: Redis
- **Recalculation**: Apenas se necessário (dados novos disponíveis)

### Infraestrutura

#### Message Queues

**RabbitMQ ou Apache Kafka**:
- **Pipeline Assíncrono**: Separar coleta de dados, análise técnica e scraping de notícias
- **Workers Especializados**:
  - `data-collector`: Busca dados OHLC a cada 15 minutos
  - `technical-analyzer`: Processa indicadores e classificação (ML Clássico)
  - `news-scraper`: Coleta e processa notícias a cada hora (Gera embeddings/vetores)
  - `insight-generator`: Gera insights quando solicitado (Disponibildiade via API para usuários)

**Benefícios:**
- Processamento paralelo
- Tolerância a falhas
- Escalabilidade horizontal

#### Bancos de Dados

1. **Vector DB** (PostgreSQL + PGVector):
   - **Dados**: Embeddings de notícias
   - **Vantagens**: Busca semântica rápida, escalável
   - **Indexação**: IVFLAT
   - **Segurança**: Implementação de RLS

3. **Cache** (Redis):
   - **Dados**: Insights, classificações, indicadores
   - **Configuração**: Cluster mode para alta disponibilidade

#### Serviço de Inference

**API REST** (FastAPI):
- **Endpoints**:
  - `GET /insight/{currency_pair}`: Retorna insight atual

**Características**:
- **Rate Limiting**: Por usuário/IP (ex: 100 req/min)
- **Caching**: Cache HTTP (Cache-Control headers)
- **Monitoring**: Prometheus + Grafana para métricas

## Estrutura do Projeto

```
forex-advisor/
├── main.py                    # Orquestrador principal
├── requirements.txt           # Dependências Python
├── Dockerfile                # Containerização
├── README.md                 # Esta documentação
├── src/
│   ├── data/
│   │   └── forex-scrapping.py    # Coleta de dados OHLC
│   ├── analysis/
│   │   └── analysis.py           # Análise técnica e classificação
│   ├── news/
│   │   └── news-scrapping.py     # Coleta de notícias
│   └── agent/
│       └── agent.py              # Geração de insights via LLM
```

# Sugestão de Arquitetura (C4 Model Level 1)