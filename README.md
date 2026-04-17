# Pipeline de Análise de Sentimento — X/Twitter Financeiro

Pipeline de coleta, anotação e análise de sentimento de publicações do X/Twitter
relacionadas ao mercado financeiro brasileiro, desenvolvido como TCC de MBA em
Inteligência Artificial e Big Data (USP São Carlos).

**Ferramenta central:** [FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR)
(Santos, Bianchi & Costa, 2023) — modelo BERT especializado para o domínio
financeiro em português do Brasil.

**Produto final:** índice de sentimento extraído das publicações, avaliado quanto
à sua coerência com eventos macroeconômicos do período.

---

## Estrutura do Projeto

```
collect-twitter-data/
│
├── .env                                    # Variáveis de ambiente (credenciais)
├── docker-compose.yml                      # Infraestrutura Docker (PostgreSQL + pgAdmin)
├── Makefile                                # Comandos por etapa do pipeline
├── pytest.ini                              # Configuração de descoberta de testes
├── requirements.txt                        # Dependências Python
│
├── config/                                 # Termos de busca para a coleta
│   ├── search_terms_monthly.json           # Períodos mensais (out/2025–fev/2026)
│   └── search_terms_historical.json        # Períodos históricos (2016–2025)
│
├── infra/                                  # Scripts de inicialização do banco
│   ├── 01-init-database.sh                 # Criação de usuário e permissões
│   └── 02-create-tables.sql               # Schema (tabelas e índices)
│
├── pipeline/
│   ├── 01_extraction/                      # Coleta via API X v2
│   │   ├── base_extractor.py               # ABC: interface para extratores de dados
│   │   ├── twitter_api.py                  # TwitterAPIExtractor(BaseExtractor)
│   │   ├── collection_log.py               # Registro de termos de busca no banco
│   │   └── __main__.py                     # Ponto de entrada: python -m pipeline.01_extraction
│   │
│   ├── 02_annotation/                      # Anotação manual (Streamlit)
│   │   ├── classifier.py                   # Interface de classificação tweet a tweet
│   │   └── app.py                          # Launcher multi-página Streamlit
│   │
│   ├── 03_eda/                             # Exploração dos dados
│   │   ├── dashboard.py                    # Distribuição de sentimentos e categorias
│   │   └── eda.py                          # Análise textual (comprimento, tokens, etc.)
│   │
│   ├── 04_preprocessing/                   # Limpeza textual
│   │   └── preprocessing_dashboard.py      # Impacto cumulativo de cada etapa no FinBERT
│   │
│   ├── 05_processing/                      # Inferência do modelo
│   │   ├── base_model.py                   # ABC: BaseSentimentAnalyzer (fetch→preprocess→predict→save)
│   │   ├── fin_bert_timbau.py              # FinBERTAnalyzer(BaseSentimentAnalyzer)
│   │   ├── fin_bert_timbau_tests.py        # Testes unitários de FinBERTAnalyzer
│   │   └── processing_dashboard.py         # Dashboard de classificação com comparação humano vs. modelo
│   │
│   └── 06_evaluation/                      # Avaliação dos resultados
│       └── metrics.py                      # Acurácia, F1, matriz de confusão
│
├── shared/                                 # Código transversal a todas as etapas
│   ├── database.py                         # DatabaseManager (conexão e queries)
│   ├── models.py                           # Dataclasses de parsing da API X
│   ├── text_cleaner.py                     # Funções de limpeza textual (URLs, emojis, stopwords, lematização)
│   └── text_cleaner_tests.py               # Testes unitários de text_cleaner
│
└── queries/
    └── analysis.sql                        # Queries SQL utilitárias para análise
```

### Arquitetura do pipeline

```
╔══════════════╦══════════════╦══════════════╗
║ 01_extraction║ 02_annotation║   03_eda     ║
║──────────────║──────────────║──────────────║
║ API X v2     ║ Classificação║ Exploração   ║
║ BaseExtractor║ manual       ║ dados brutos ║
║ (abstrato)   ║ (Streamlit)  ║ e anotados   ║
╚══════╤═══════╩══════════════╩══════╤═══════╝
       │ fluxo de dados              │ informa decisões de limpeza
       ▼                             ▼
╔══════════════╦══════════════╦══════════════╗
║04_preprocess ║ 05_processing║06_evaluation ║
║──────────────║──────────────║──────────────║
║ Limpeza      ║ FinBERT-     ║ Acurácia     ║
║ textual      ║ PT-BR        ║ F1 · matriz  ║
║              ║ inferência   ║ confusão     ║
╚══════════════╩══════════════╩══════════════╝

─────────── shared/database.py · shared/models.py · infra/ · config/ ───────────
```

---

## Pré-requisitos

- [Docker](https://www.docker.com/) e Docker Compose instalados
- Python 3.10+
- Credenciais da [API do X v2](https://developer.x.com/) (Bearer Token)

---

## 1. Configuração Inicial

### 1.1. Variáveis de Ambiente

Edite o arquivo `.env` na raiz do projeto:

```dotenv
# PostgreSQL
POSTGRES_DB=twitter_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=sua_senha_aqui
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Usuário da aplicação
POSTGRES_TWITTER_USER=twitter_user
POSTGRES_TWITTER_PASSWORD=sua_senha_aqui

# pgAdmin (interface gráfica opcional)
PGADMIN_DEFAULT_EMAIL=admin@twitter.com
PGADMIN_DEFAULT_PASSWORD=admin123
PGADMIN_PORT=5050

# API X v2
TWITTER_ACCESS_TOKEN=seu_bearer_token_aqui
```

### 1.2. Permissões do Script de Inicialização

```bash
chmod +x infra/01-init-database.sh
```

---

## 2. Infraestrutura Docker

```bash
make db-up      # sobe PostgreSQL + pgAdmin
make db-down    # para os containers
```

Ao subir, o Docker inicializa automaticamente:
1. PostgreSQL 15 (porta `5432`)
2. pgAdmin 4 (porta `5050`)
3. `infra/01-init-database.sh` — cria o usuário `twitter_user` e configura permissões
4. `infra/02-create-tables.sql` — cria as tabelas `tweets`, `tweets_classification` e `collection_log`

---

## 3. Ambiente Python

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## 4. Executando o Pipeline

Todos os comandos são executados a partir da raiz do projeto via `make`:

| Comando | Descrição | Status |
|---|---|---|
| `make collect` | Coleta tweets via API X v2 | Implementado |
| `make annotate` | Anotação manual (Streamlit) | Implementado |
| `make eda` | Dashboard de exploração dos dados | Implementado |
| `make preprocess` | Dashboard de impacto do pré-processamento | Implementado |
| `make process` | Dashboard de classificação com FinBERT-PT-BR | Implementado |
| `make evaluate` | Métricas de avaliação | A implementar |

### 4.1. Registrar os Termos de Busca

Antes da primeira coleta, popula `collection_log` com os períodos e usuários:

```bash
python -m pipeline.01_extraction.collection_log
```

Cada entrada em `config/search_terms_monthly.json` define:
- `x_user_id` — ID do usuário no X (ex: InfoMoney = `59773459`)
- `from_date_time` — início do período (ISO 8601, UTC)
- `to_date_time` — fim do período (ISO 8601, UTC)

### 4.2. Coleta

```bash
make collect
```

1. Busca logs com status `pending` em `collection_log`
2. Chama a API X v2 via `TwitterAPIExtractor` com paginação automática
3. Insere tweets em `tweets`
4. Atualiza o log com `start_time`, `end_time` e status (`completed` / `partially_completed`)

#### Extensibilidade

Novas fontes de dados implementam `BaseExtractor`:

```python
from pipeline.01_extraction.base_extractor import BaseExtractor

class MyExtractor(BaseExtractor):
    def fetch(self, user_id: str, from_dt: str, to_dt: str) -> list[dict]:
        ...
```

### 4.3. Anotação Manual

```bash
make annotate
# Acesse http://localhost:8501
```

- Exibe tweets não classificados um a um
- Pergunta se o tweet é **notícia financeira** (Sim / Não)
- Para tweets financeiros, solicita o **sentimento** (Positivo / Neutro / Negativo)
- Permite adicionar justificativa para cada classificação
- Salva em `tweets_classification` com `classificator = "Humano"`

### 4.4. Exploração dos Dados (EDA)

```bash
make eda
# Acesse http://localhost:8501
```

Dashboard com:
- Distribuição de sentimentos e categorias (financeiro / não financeiro)
- Distribuição temporal e por veículo
- Análise de comprimento dos textos (caracteres e tokens)
- Campos ausentes e publicações atípicas

### 4.5. Pré-processamento

```bash
make preprocess
# Acesse http://localhost:8501
```

Dashboard interativo que aplica as etapas de limpeza de forma **cumulativa** (URL → Emojis → Menções → Hashtags → Espaços → Caixa baixa → Stopwords → Lematização) e executa o FinBERT-PT-BR após cada etapa, permitindo avaliar o impacto de cada passo na classificação final.

As funções de limpeza residem em `shared/text_cleaner.py` e são compartilhadas entre `04_preprocessing` e `05_processing`.

### 4.6. Processamento (Inferência)

```bash
make process
# Acesse http://localhost:8501
```

Dashboard de classificação que:
- Seleciona o algoritmo (atualmente FinBERT-PT-BR; arquitetura extensível via `BaseSentimentAnalyzer`)
- Classifica tweets financeiros com anotação humana existente
- Compara sentimento do modelo com o sentimento humano tweet a tweet
- Exibe métricas de concordância e permite salvar as classificações no banco

#### Extensibilidade

Novos modelos implementam `BaseSentimentAnalyzer`:

```python
from pipeline.05_processing.base_model import BaseSentimentAnalyzer

class MyModelAnalyzer(BaseSentimentAnalyzer):
    classificator = "MyModel"

    def load_model(self): ...
    def preprocess(self, text: str) -> str: ...
    def predict_text(self, text: str, batch_size: int) -> list[dict]: ...
    def run(self) -> pd.DataFrame: ...
```

---

## 5. Testes

Os testes ficam junto ao módulo que testam (co-located), dentro de `pipeline/` ou `shared/`.

```bash
python -m pytest          # todos os testes
python -m pytest -v       # com detalhes por teste
```

| Arquivo | Módulo testado | Cobertura |
|---|---|---|
| `shared/text_cleaner_tests.py` | funções de `text_cleaner` | URLs, emojis, menções, hashtags, espaços, stopwords, lematização |
| `pipeline/05_processing/fin_bert_timbau_tests.py` | `FinBERTAnalyzer` | atributos de classe, `preprocess`, `normalize_label`, `predict_text`, `run`, `save` |

A descoberta é configurada em `pytest.ini` na raiz: coleta arquivos `test_*.py` e `*_tests.py` dentro de `pipeline/` e `shared/`.

---

## 6. Banco de Dados

### Tabela `tweets`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | SERIAL | Chave primária |
| `tweet_id` | VARCHAR(50) | ID único do tweet no X |
| `username` | VARCHAR(100) | ID do autor (user_id da API) |
| `note_tweet` | TEXT | Texto completo (tweets longos) |
| `created_at` | TIMESTAMPTZ | Data/hora de publicação |
| `likes` | INTEGER | Número de curtidas |
| `hashtags` | JSONB | Lista de hashtags |
| `tweet` | JSONB | Objeto completo retornado pela API |
| `sentiment` | VARCHAR(8) | `positivo`, `negativo` ou `neutro` |
| `is_finance_news` | INTEGER | `1` = financeiro, `0` = não financeiro |

### Tabela `tweets_classification`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | SERIAL | Chave primária |
| `tweet_id` | INTEGER | FK → `tweets.id` |
| `sentiment` | VARCHAR(8) | Sentimento classificado |
| `why_sentiment` | TEXT | Justificativa do sentimento |
| `is_finance_news` | INTEGER | Classificação financeiro/não financeiro |
| `why_is_finance_news` | TEXT | Justificativa da classificação |
| `classificator` | VARCHAR(100) | `"Humano"` ou `"FinBERT-PT-BR"` |
| `score` | REAL | Score de confiança do modelo (se aplicável) |

### Tabela `collection_log`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | SERIAL | Chave primária |
| `search_term` | JSONB | `{x_user_id, from_date_time, to_date_time}` |
| `tweets_collected` | INTEGER | Quantidade de tweets inseridos |
| `start_time` | TIMESTAMP | Início da coleta |
| `end_time` | TIMESTAMP | Fim da coleta |
| `status` | VARCHAR(20) | `pending`, `completed`, `partially_completed` |
| `error_message` | TEXT | Mensagem de erro (se houver) |

---

## 7. Configuração do pgAdmin

Acesse em `http://localhost:5050` com as credenciais definidas em `.env`.

**Conexão Superusuário:**
```
Host: postgres | Porta: 5432 | Banco: postgres | Usuário: postgres
```

**Conexão Aplicação:**
```
Host: postgres | Porta: 5432 | Banco: twitter_db | Usuário: twitter_user
```

---

## 8. Fontes de Dados

| Conta X | ID | Volume no dataset |
|---|---|---|
| InfoMoney | `59773459` | ~88,5% |
| InvestingBrasil | `51150679` | ~11,5% |

Dataset total: ~3.563 tweets (outubro/2025 a fevereiro/2026).
Coleta via `GET /2/users/{id}/tweets` (API X v2, Bearer Token).

---

## 9. Repositórios

- **TCC (LaTeX):** https://github.com/toninlopes/usp-tcc-mba-ia-bigdata
- **Pipeline:** https://github.com/toninlopes/usp-tcc-pipeline-mba-ia-bigdata
