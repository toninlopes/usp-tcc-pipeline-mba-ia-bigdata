# Coletor de Notícias Financeiras — X/Twitter

Pipeline de coleta, armazenamento e classificação de tweets financeiros, usando a API do X (Twitter) v2, PostgreSQL e uma interface Streamlit para classificação manual e análise de dados.

---

## Estrutura do Projeto

```
collect-twitter-data/
├── .env                                        # Variáveis de ambiente (credenciais e configurações)
├── docker-compose.yml                          # Infraestrutura Docker (PostgreSQL + pgAdmin)
├── init-scripts/
│   ├── 01-init-database.sh                     # Criação de usuário e permissões no banco
│   └── 02-create-tables.sql                    # Schema do banco (tabelas, índices)
├── json_data/
│   ├── search_tems.json                        # Termos de busca mensais (out–dez 2025)
│   └── all_search.json                         # Termos de busca históricos (2016–2025)
├── python_app/
│   ├── __main__.py                             # Ponto de entrada principal (coleta de tweets)
│   ├── x_tweets.py                             # Cliente da API X v2
│   ├── parse_tweet.py                          # Data classes para parsing do JSON da API
│   ├── database.py                             # Gerenciador de conexão e queries PostgreSQL
│   ├── collection_log.py                       # Registro de termos de busca no banco
│   ├── classifier_app.py                       # Launcher do app Streamlit
│   ├── check_db_connection.py                  # Teste de conectividade com o banco
│   ├── requirements.txt                        # Dependências Python
│   └── classifier/
│       ├── classifier.py                       # Interface de classificação manual (Streamlit)
│       └── dashboard.py                        # Dashboard de análise (Streamlit)
└── QUERIES                                     # Queries SQL utilitárias para análise
```

---

## Pré-requisitos

- [Docker](https://www.docker.com/) e Docker Compose instalados
- Python 3.9+
- Credenciais da [API do X v2](https://developer.x.com/) (Bearer Token)
- Conta no X com acesso à API de pesquisa de tweets por usuário

---

## 1. Configuração Inicial

### 1.1. Variáveis de Ambiente

Edite o arquivo `.env` na raiz do projeto com suas credenciais:

```dotenv
# PostgreSQL
POSTGRES_DB=twitter_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=sua_senha_aqui
POSTGRES_PORT=5432

# Usuário da aplicação
POSTGRES_TWITTER_USER=twitter_user
POSTGRES_TWITTER_PASSWORD=sua_senha_aqui

# PgAdmin (opcional - interface gráfica)
PGADMIN_DEFAULT_EMAIL=admin@twitter.com
PGADMIN_DEFAULT_PASSWORD=admin123
PGADMIN_PORT=5050

# Twitter API credentials (se for usar API oficial)
TWITTER_ACCESS_TOKEN=seu_beare_token_aqui
```

### 1.2. Permissões do Script de Inicialização

```bash
chmod +x init-scripts/01-init-database.sh
```

---

## 2. Infraestrutura Docker

### Subir os containers

```bash
docker-compose up -d
```

Isso inicializa automaticamente:
1. PostgreSQL 15 (container `twitter-postgres`, porta 5432)
2. pgAdmin 4 (container `twitter-pgadmin`, porta 5050)
3. Script `01-init-database.sh` — cria o usuário `twitter_user` e configura permissões
4. Script `02-create-tables.sql` — cria as tabelas `tweets`, `tweets_classification` e `collection_log`

### Outros comandos Docker

```bash
# Parar os containers
docker-compose down

# Parar e remover volumes (atenção: apaga todos os dados)
docker-compose down -v

# Acompanhar logs do PostgreSQL
docker-compose logs -f postgres
```

### Teste de Conexão

```bash
# Superusuário
docker exec -it twitter-postgres psql -U postgres -c "SELECT current_user, current_database();"

# Usuário da aplicação
docker exec -it twitter-postgres psql -U twitter_user -d twitter_db -c "SELECT current_user, current_database();"

# Via Python
python ./python_app/check_db_connection.py
```

---

## 3. Ambiente Python

```bash
# Criar e ativar o virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r python_app/requirements.txt
```

---

## 4. Pipeline de Coleta

### 4.1. Popular o Banco com Termos de Busca

Antes de coletar tweets, registre os termos de busca (períodos e usuários alvo) na tabela `collection_log`:

```bash
python -m python_app.collection_log
```

Os termos de busca são lidos dos arquivos JSON em `json_data/`. Cada entrada define:
- `x_user_id` — ID do usuário no X a ser monitorado (ex: InfoMoney = `59773459`)
- `from_date_time` — início do período de coleta (ISO 8601, UTC)
- `to_date_time` — fim do período de coleta (ISO 8601, UTC)

### 4.2. Executar a Coleta de Tweets

```bash
python -m python_app
```

O script `__main__.py`:
1. Busca todos os logs com status `pending` no banco
2. Para cada log pendente, chama a API do X v2 para o usuário e período definidos
3. Faz a paginação automática dos resultados (`next_token`)
4. Filtra e descarta tweets sem conteúdo de texto
5. Insere os tweets na tabela `tweets`
6. Atualiza o log com `start_time`, `end_time`, contagem de tweets e status (`completed` ou `partially_completed`)

---

## 5. Classificação de Tweets

### 5.1. Interface de Classificação Manual

```bash
streamlit run python_app/classifier_app.py
```

Acesse em `http://localhost:8501`.

O app possui duas páginas:

**Classificador Manual**
- Exibe tweets não classificados um a um
- Pergunta se o tweet é uma **notícia financeira** (Sim / Não)
- Para tweets financeiros, solicita o **sentimento** (Positivo / Neutro / Negativo)
- Permite adicionar uma justificativa para cada classificação
- Exibe classificações anteriores (humana e/ou por modelos de IA)
- Salva os resultados na tabela `tweets_classification` com o classificador `"Humano"`

**Dashboard de Análise**
- Gráfico de pizza: distribuição de tweets financeiros vs. não financeiros vs. não classificados
- Gráfico de pizza: distribuição de sentimentos (apenas tweets financeiros)
- Estatísticas resumidas por categoria

---

## 6. Fluxo do Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                     FASE 1: COLETA                      │
│                                                         │
│  json_data/*.json  →  collection_log.py                 │
│  (termos de busca)     (popula collection_log           │
│                         com status "pending")           │
│                              │                          │
│                              ▼                          │
│                        __main__.py                      │
│                    (lê logs "pending")                  │
│                              │                          │
│                              ▼                          │
│                        x_tweets.py                      │
│                    (API X v2 + paginação)                │
│                              │                          │
│                              ▼                          │
│                       parse_tweet.py                    │
│                    (parsing do JSON da API)              │
│                              │                          │
│                              ▼                          │
│                        database.py                      │
│                 (insert na tabela "tweets")              │
│                 (update em "collection_log")             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  FASE 2: CLASSIFICAÇÃO                  │
│                                                         │
│                     classifier.py                       │
│               (Streamlit — classificação manual)        │
│                              │                          │
│                              ▼                          │
│                        database.py                      │
│            (insert em "tweets_classification"           │
│             com classificador "Humano" ou IA)           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    FASE 3: ANÁLISE                      │
│                                                         │
│                      dashboard.py                       │
│            (Streamlit — visualizações e métricas)       │
│                              │                          │
│                              ▼                          │
│                  Queries SQL (arquivo QUERIES)          │
│            (filtragem por hashtag, agregação            │
│             por modo estatístico, etc.)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Schema do Banco de Dados

### Tabela `tweets`
Armazena os tweets coletados da API.

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | SERIAL | Chave primária |
| `tweet_id` | VARCHAR(50) | ID único do tweet no X |
| `username` | VARCHAR(100) | Nome de usuário do autor |
| `note_tweet` | TEXT | Texto completo (tweets longos) |
| `created_at` | TIMESTAMPTZ | Data/hora de publicação |
| `likes` | INTEGER | Número de curtidas |
| `hashtags` | JSONB | Lista de hashtags |
| `tweet` | JSONB | Objeto completo retornado pela API |
| `sentiment` | VARCHAR(8) | `positivo`, `negativo` ou `neutro` |
| `is_finance_news` | INTEGER | `1` = financeiro, `0` = não financeiro |

### Tabela `tweets_classification`
Armazena as classificações (humanas e por IA) para cada tweet.

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | SERIAL | Chave primária |
| `tweet_id` | INTEGER | FK → `tweets.id` |
| `sentiment` | VARCHAR(8) | Sentimento classificado |
| `why_sentiment` | TEXT | Justificativa do sentimento |
| `is_finance_news` | INTEGER | Classificação financeiro/não financeiro |
| `why_is_finance_news` | TEXT | Justificativa da classificação |
| `classificator` | VARCHAR(100) | Ex: `"Humano"`, `"Sonnet 4.6"` |

### Tabela `collection_log`
Registra o histórico de coletas executadas.

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

## 8. Configuração do pgAdmin

Acesse em `http://localhost:5050` com as credenciais definidas em `.env`.

**Conexão Superusuário:**
```
Host: postgres
Porta: 5432
Banco: postgres
Usuário: postgres
Senha: $POSTGRES_PASSWORD
```

**Conexão Aplicação:**
```
Host: postgres
Porta: 5432
Banco: twitter_db
Usuário: twitter_user
Senha: $POSTGRES_TWITTER_PASSWORD
```

---

## 9. Credenciais e Permissões

| Perfil | Usuário | Banco | Privilégios |
|---|---|---|---|
| Superusuário | `postgres` | `postgres` | Todos |
| Aplicação | `twitter_user` | `twitter_db` | SELECT, INSERT, UPDATE, DELETE |
