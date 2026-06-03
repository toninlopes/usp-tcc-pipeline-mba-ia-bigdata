# Pipeline de Análise de Sentimento — X/Twitter Financeiro

Pipeline de coleta, anotação e análise de sentimento de publicações do X/Twitter
relacionadas ao mercado financeiro brasileiro, desenvolvido como TCC de MBA em
Inteligência Artificial e Big Data (USP São Carlos).

**Modelos avaliados:**
- **BERT:** [FinBERT-PT-BR](https://huggingface.co/lucas-leme/FinBERT-PT-BR) (Santos, Bianchi & Costa, 2023) — modelo BERT especializado para o domínio financeiro em português do Brasil; [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) (Souza et al., 2020) — BERT pré-treinado em português, com fine-tuning sobre o dataset anotado.
- **Léxico:** [SentiLex-PT](http://l2f.inesc-id.pt/wiki/index.php/SentiLex-PT) (Silva et al., 2012) — léxico de sentimento para o português; [OpLexicon](https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/) (Souza & Vieira, 2012) — léxico de opinião para o português brasileiro.

**Produto final:** índice de sentimento extraído das publicações, avaliado quanto
à sua coerência com eventos macroeconômicos do período.

---

## Estrutura do Projeto

```
collect-twitter-data/
│
├── .env                                      # Variáveis de ambiente (credenciais)
├── docker-compose.yml                        # Infraestrutura Docker (PostgreSQL + pgAdmin)
├── Makefile                                  # Comandos por etapa do pipeline
├── pytest.ini                                # Configuração de descoberta de testes
├── requirements.txt                          # Dependências Python
│
├── app/                                      # Todo o código Python
│   ├── core/                                 # Lógica de negócio (sem UI)
│   │   ├── extraction/                       # Coleta via API X v2
│   │   │   ├── base_extractor.py             # ABC: interface para extratores de dados
│   │   │   ├── twitter_api.py                # TwitterAPIExtractor(BaseExtractor)
│   │   │   └── collection_log.py             # Registro de coletas no banco
│   │   ├── classification/                   # Inferência de sentimento
│   │   │   ├── base_analyzer.py              # ABC: contrato público dos analisadores
│   │   │   ├── bert/                         # Modelos baseados em BERT
│   │   │   │   ├── bert_analyzer.py          # BertSentimentAnalyzer (base)
│   │   │   │   ├── finbert_ptbr.py           # FinBertPTBRAnalyzer
│   │   │   │   └── bert_timbau.py            # BERTimbauAnalyzer
│   │   │   └── lexicon/                      # Modelos baseados em léxico
│   │   │       ├── lexicon_analyzer.py       # LexiconSentimentAnalyzer (base)
│   │   │       ├── senti_lex.py              # SentiLexAnalyzer
│   │   │       └── op_lexicon.py             # OpLexiconAnalyzer
│   │   ├── fine_tuning/                      # Fine-tuning do BERTimbau
│   │   │   └── bert_timbau_fine_tuner.py     # K-Fold estratificado, WeightedTrainer
│   │   └── evaluation/
│   │       └── metrics.py                    # Acurácia, F1, matriz de confusão
│   │
│   ├── dashboard/                            # UI Streamlit (sem lógica de negócio)
│   │   ├── app.py                            # Entrypoint multi-página
│   │   └── pages/                            # Uma página por etapa do pipeline
│   │       ├── annotation.py                 # Anotação manual tweet a tweet
│   │       ├── exploration.py                # Exploração e analytics dos dados
│   │       ├── dataset_split.py              # Particionamento train/test/fold
│   │       ├── fine_tuning.py                # Fine-tuning do BERTimbau (dashboard)
│   │       ├── classification.py             # Inferência dos modelos com pré-processamento
│   │       └── evaluation.py                 # Métricas de avaliação
│   │
│   └── shared/                               # Código transversal
│       ├── db/                               # Acesso ao banco de dados
│       │   ├── database.py                   # DatabaseManager (pool de conexões)
│       │   ├── tweets.py                     # TweetsRepository
│       │   ├── classification.py             # ClassificationRepository
│       │   ├── collection_log.py             # CollectionLogRepository
│       │   └── dataset_split.py              # DatasetSplitRepository
│       ├── schemas.py                        # Dataclasses de parsing da API X v2
│       └── text_cleaner.py                   # Funções de limpeza textual
│
├── config/                                   # Termos de busca para a coleta
│   ├── search_terms_monthly.json             # Períodos mensais (out/2025–fev/2026)
│   └── search_terms_historical.json          # Períodos históricos (2016–2025)
│
├── data/                                     # Léxicos de sentimento (download automático)
│
├── infra/                                    # Scripts de inicialização do banco
│   ├── 01-init-database.sh                   # Criação de usuário e permissões
│   └── 02-create-tables.sql                  # Schema (tabelas e índices)
│
├── models/                                   # Modelos fine-tuned (gerados localmente)
│   └── bert-timbau-sentiment/                # Produzido por bert_timbau_fine_tuner.py
│
└── queries/                                  # SQL utilitário
    └── queries.sql
```

### Arquitetura do pipeline

```
╔══════════════╦═══════════════════╦══════════════════════╗
║  extraction  ║    annotation     ║      exploration     ║
║──────────────║───────────────────║──────────────────────║
║ API X v2     ║ Classificação     ║ Exploração, analytics║
║ BaseExtractor║ manual (Streamlit)║ distribuição e EDA   ║
╚══════╤═══════╩═══════════════════╩══════════╤═══════════╝
       │ fluxo de dados                        │ informa decisões
       ▼                                       ▼
╔══════════════╦══════════════╦════════════════╦══════════════╗
║dataset_split ║ fine_tuning  ║ classification ║  evaluation  ║
║──────────────║──────────────║────────────────║──────────────║
║ Hold-out +   ║ BERTimbau    ║ FinBERT·BERT   ║ Acurácia     ║
║ K-Fold       ║ K-Fold       ║ Léxico +       ║ F1 · matriz  ║
║ estratificado║ estratificado║ pré-proc. opt. ║ confusão     ║
╚══════════════╩══════════════╩════════════════╩══════════════╝

──────────── app/shared/db/ · app/shared/schemas.py · infra/ · config/ ────────────
```

---

## Pré-requisitos

- [Docker](https://www.docker.com/) e Docker Compose instalados
- Python 3.9+
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

Todos os comandos são executados a partir da raiz do projeto via `make`.
`make dashboard` abre o aplicativo Streamlit multi-página em `http://localhost:8501`
— a navegação entre etapas é feita pelo menu lateral.

| Comando | Descrição |
|---|---|
| `make collect` | Coleta tweets via API X v2 |
| `make dashboard` | Abre o dashboard Streamlit multi-página (`http://localhost:8501`) |
| `make evaluate` | Executa métricas de avaliação (CLI) |
| `make db-up` | Sobe PostgreSQL + pgAdmin via Docker |
| `make db-down` | Para os containers |

### 4.1. Registrar os Termos de Busca

Antes da primeira coleta, popula `collection_log` com os períodos e usuários:

```bash
PYTHONPATH=. python -m app.core.extraction.collection_log
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
from app.core.extraction.base_extractor import BaseExtractor

class MyExtractor(BaseExtractor):
    def fetch(self, user_id: str, from_dt: str, to_dt: str) -> list[dict]:
        ...
```

### 4.3. Anotação Manual

```bash
make dashboard
# Acesse http://localhost:8501
```

- Exibe tweets não classificados um a um
- Pergunta se o tweet é **notícia financeira** (Sim / Não)
- Para tweets financeiros, solicita o **sentimento** (Positivo / Neutro / Negativo)
- Permite adicionar justificativa para cada classificação
- Salva em `tweets_classification` com `classificator = "Humano"`

### 4.4. Exploração dos Dados

```bash
make dashboard
# Navegue até "🔍 Exploração dos Dados" no menu lateral
```

Dashboard com:
- Dimensões do dataset: total de tweets, veículos, período, financeiros e não financeiros
- Distribuição de sentimentos (gráfico de barras) e financeiro vs. não financeiro (pizza)
- Distribuição temporal e por veículo
- Análise de comprimento dos textos (caracteres e tokens)
- Campos ausentes e publicações atípicas

### 4.6. Split do Dataset

```bash
make dashboard
# Navegue até "✂️ Split do Dataset" no menu lateral
```

Particiona os tweets com anotação humana em:
- **Hold-out de teste** (~15%) — conjunto fixo usado na avaliação comparativa entre todos os modelos.
- **Treino com K-Fold estratificado** (restante) — dividido em 4 folds por sentimento, usado no fine-tuning do BERTimbau.

A operação é idempotente: se o split já existir, exibe o resumo atual. Use a opção
"Re-gerar split" para apagar e recriar. Gerenciado por `DatasetSplitRepository`
(`app/shared/db/dataset_split.py`).

### 4.7. Fine-tuning do BERTimbau

Pré-requisito para usar o `BERTimbauAnalyzer` no passo de processamento.
Requer o split atribuído (passo 4.6) e mínimo ~300 tweets anotados.

**Via dashboard (recomendado):**

```bash
make dashboard
# Navegue até "🧠 Fine-tuning BERTimbau" no menu lateral
```

A página permite configurar hiperparâmetros (learning rate, batch size, épocas, etc.),
iniciar o treinamento e acompanhar o progresso em tempo real por fold/época.
Registra um histórico de execuções em `models/bert-timbau-sentiment/training_runs.json`.

**Via CLI:**

```bash
PYTHONPATH=. python -m app.core.fine_tuning.bert_timbau_fine_tuner
```

O fine-tuner executa K-Fold estratificado (4 folds) usando o particionamento
persistido na tabela `dataset_split` pelo `DatasetSplitRepository`. O modelo do
melhor fold (maior F1 macro na validação) é salvo em `models/bert-timbau-sentiment/`.
Usa `WeightedTrainer` para lidar com desbalanceamento de classes e
`EarlyStoppingCallback` com paciência de 2 épocas.

### 4.8. Classificação (Inferência)

```bash
make dashboard
# Navegue até "🤖 Classificação" no menu lateral
```

Dashboard de classificação que:
- Seleciona o algoritmo entre os disponíveis: **FinBERT-PT-BR**, **BERTimbau**, **SentiLex-PT**, **OpLexicon**
- Permite configurar etapas de pré-processamento textual via checkboxes (remoção de URLs,
  menções, hashtags, emojis, pontuação de borda, stopwords, lematização, etc.)
- Salva e restaura automaticamente a melhor configuração de pré-processamento por modelo
  em `models/best_preprocessing.json` (apenas quando a concordância melhora)
- Classifica o conjunto hold-out (`split='test'`) do `DatasetSplitRepository`
- Sempre atualiza o banco: DELETE + INSERT por tweet/classificador
- Exibe a lista de tweets classificados com sentimento humano, sentimento do modelo e concordância

> **Nota:** o BERTimbau requer fine-tuning prévio (passo 4.7).

#### Extensibilidade

Novos modelos implementam `BaseSentimentAnalyzer`:

```python
from app.core.classification.base_analyzer import BaseSentimentAnalyzer

class MyModelAnalyzer(BaseSentimentAnalyzer):
    classificator = "MyModel"

    def preprocess(self, text: str) -> str: ...
    def predict(self, text: str) -> Tuple[str, float]: ...
    def run(self) -> pd.DataFrame: ...
```

### 4.9. Avaliação

```bash
make evaluate
```

Executa as métricas de avaliação em modo CLI comparando as classificações de cada
modelo contra o gold standard humano no conjunto hold-out (`split='test'`):
acurácia, F1 macro, F1 por classe e matriz de confusão.

---

## 5. Testes

Os testes ficam co-localizados com o módulo que testam (`*_tests.py`), dentro de `app/`.

```bash
PYTHONPATH=. python -m pytest app/ -v                              # todos os testes
PYTHONPATH=. python -m pytest app/shared/ -v                       # apenas shared
PYTHONPATH=. python -m pytest app/core/ -v                         # apenas core
PYTHONPATH=. python -m pytest app/core/classification/bert/ -v     # apenas modelos BERT
PYTHONPATH=. python -m pytest app/core/classification/lexicon/ -v  # apenas modelos léxicos
```

A descoberta é configurada em `pytest.ini` na raiz: coleta arquivos `test_*.py` e `*_tests.py` dentro de `app/`.

O pre-commit hook (`.git/hooks/pre-commit`) bloqueia commits automaticamente caso qualquer
teste falhe ou a cobertura fique abaixo de 80%. A cobertura é configurada via `.coveragerc`,
que exclui `app/dashboard/` (páginas Streamlit, não testáveis unitariamente) e o fine-tuner
(`bert_timbau_fine_tuner.py`, que requer GPU e downloads de modelo).

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
| `classificator` | VARCHAR(100) | Origem: `"Humano"`, `"FinBERT-PT-BR"`, `"BERTimbau"`, `"SentiLex-PT"`, `"OpLexicon"` |
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

### Tabela `dataset_split`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | SERIAL | Chave primária |
| `tweet_id` | INTEGER | FK → `tweets.id` |
| `split` | VARCHAR(10) | `'train'` ou `'test'` |
| `fold` | SMALLINT | Fold do K-Fold (1–4) para `train`; `NULL` para hold-out (`test`) |

Restrições: `tweet_id` único; fold `NULL` quando `split='test'`, entre 1 e 4 quando `split='train'`.

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
