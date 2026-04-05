# CLAUDE.md — usp-tcc-pipeline-mba-ia-bigdata

Arquivo de contexto lido automaticamente pelo Claude Code a cada sessão.
Não remover nem renomear.

---

## 1. O que é este projeto

Pipeline de coleta, anotação e análise de sentimento de publicações do
X/Twitter relacionadas ao mercado financeiro brasileiro, desenvolvido como
TCC de MBA em Inteligência Artificial e Big Data (USP São Carlos).

**Ferramenta central:** FinBERT-PT-BR (Santos, Bianchi & Costa, 2023) —
modelo BERT especializado para o domínio financeiro em português do Brasil.

**Produto final:** um índice de sentimento extraído das publicações,
avaliado quanto à sua coerência com eventos macroeconômicos do período.

### Escopo estrito — nunca violar

- O objetivo é extrair e analisar **polaridade** (positivo / negativo / neutro)
- NÃO inclui modelagem preditiva de índices de mercado (IBOV, IDIV, IFIX, IBrX, etc.)
- NÃO inclui construção de estratégias de investimento
- Ao mencionar B3 ou mercado financeiro, tratar como **contexto**, não como variável a prever

---

## 2. Problemas da estrutura atual

### 2.1 Tudo misturado em `python_app/` sem hierarquia de responsabilidades

```
python_app/          ← módulo único com responsabilidades misturadas
├── __main__.py      # coleta
├── x_tweets.py      # cliente API
├── parse_tweet.py   # parsing / dataclasses
├── database.py      # banco de dados
├── collection_log.py
├── classifier_app.py
└── classifier/
    ├── classifier.py   # anotação manual
    └── dashboard.py    # EDA  ← propósito diferente, mesmo diretório
```

### 2.2 `classifier/` mistura anotação com análise exploratória

`classifier.py` é uma ferramenta de **anotação** (produz dados rotulados).
`dashboard.py` é uma ferramenta de **observação** (consome dados de qualquer etapa).
São etapas conceitualmente distintas acopladas no mesmo diretório.

### 2.3 Sem abstração para múltiplas fontes de dados

A extração está acoplada à API X v2. Não há interface que permita adicionar
web scraping ou outra fonte sem modificar o código existente.

### 2.4 Pré-processamento, processamento e avaliação não têm lugar definido

Os módulos mais importantes para o TCC (limpeza textual, inferência com
FinBERT-PT-BR e avaliação dos resultados) ainda não existem e não têm
estrutura de diretório que indique onde implementá-los.

### 2.5 `__main__.py` é o único ponto de entrada

Não é possível executar etapas individuais do pipeline de forma isolada.

---

## 3. Arquitetura proposta do pipeline

A EDA precede o pré-processamento: os dados brutos e já anotados precisam
ser explorados antes de definir o que e como limpar.

```
╔══════════════════════════════════════════════════════════════════════╗
║                  PIPELINE DE ANÁLISE DE SENTIMENTO                   ║
╠══════════════╦══════════════╦══════════════╗                         ║
║ 01_extraction║ 02_annotation║   03_eda     ║                         ║
║──────────────║──────────────║──────────────║                         ║
║ API X v2     ║ Classificação║ Exploração   ║                         ║
║ BaseExtractor║ manual       ║ dados brutos ║                         ║
║ (abstrato)   ║ (Streamlit)  ║ e anotados   ║                         ║
╚══════╤═══════╩══════════════╩══════╤═══════╝                         ║
       │           fluxo             │ informa decisões                 ║
       ▼           de dados          ▼ de limpeza                      ║
╔══════════════╦══════════════╦══════════════╗                         ║
║04_preprocess ║ 05_processing║06_evaluation ║                         ║
║──────────────║──────────────║──────────────║                         ║
║ Limpeza      ║ FinBERT-     ║ Acurácia     ║                         ║
║ textual      ║ PT-BR        ║ F1 · matriz  ║                         ║
║              ║ inferência   ║ confusão     ║                         ║
╚══════════════╩══════════════╩══════════════╝                         ║
                                                                        ║
────────────────────────── infraestrutura ─────────────────────────────║
  shared/database.py · shared/models.py · infra/ · config/             ║
═══════════════════════════════════════════════════════════════════════╝
```

---

## 4. Estrutura de arquivos proposta (destino da migração)

```
usp-tcc-pipeline/
│
├── CLAUDE.md                          ← este arquivo
├── .env
├── docker-compose.yml
├── Makefile                           ← novo: comandos por etapa
├── requirements.txt                   ← era: python_app/requirements.txt
│
├── config/                            ← era: json_data/
│   ├── search_terms_monthly.json      ← era: json_data/search_tems.json
│   └── search_terms_historical.json   ← era: json_data/all_search.json
│
├── infra/                             ← era: init-scripts/ na raiz
│   ├── 01-init-database.sh
│   └── 02-create-tables.sql
│
├── pipeline/
│   │
│   ├── 01_extraction/                 ← era: python_app/ (parte)
│   │   ├── __init__.py
│   │   ├── base_extractor.py          ← NOVO: classe abstrata (ABC)
│   │   ├── twitter_api.py             ← era: x_tweets.py + parse_tweet.py
│   │   └── collection_log.py          ← era: collection_log.py
│   │
│   ├── 02_annotation/                 ← era: python_app/classifier/
│   │   ├── __init__.py
│   │   ├── classifier.py              ← era: classifier/classifier.py
│   │   └── app.py                     ← era: classifier_app.py
│   │
│   ├── 03_eda/                        ← era: python_app/classifier/dashboard.py
│   │   ├── __init__.py                   precede preprocessing: entender os dados
│   │   └── dashboard.py                  brutos informa as decisões de limpeza
│   │
│   ├── 04_preprocessing/              ← NOVO (a implementar)
│   │   ├── __init__.py
│   │   └── text_cleaner.py
│   │
│   ├── 05_processing/                 ← NOVO (a implementar)
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py
│   │
│   └── 06_evaluation/                 ← NOVO (a implementar)
│       ├── __init__.py
│       └── metrics.py
│
├── shared/                            ← NOVO: código transversal a todas as etapas
│   ├── __init__.py
│   ├── database.py                    ← era: python_app/database.py
│   └── models.py                      ← era: python_app/parse_tweet.py (dataclasses)
│
└── queries/                           ← era: QUERIES (arquivo sem extensão)
    └── analysis.sql
```

---

## 5. Mapeamento de migração (origem → destino)

| Arquivo atual                          | Destino proposto                          |
|----------------------------------------|-------------------------------------------|
| `python_app/__main__.py`               | `pipeline/01_extraction/__main__.py`      |
| `python_app/x_tweets.py`              | `pipeline/01_extraction/twitter_api.py`   |
| `python_app/parse_tweet.py`           | `shared/models.py`                        |
| `python_app/database.py`              | `shared/database.py`                      |
| `python_app/collection_log.py`        | `pipeline/01_extraction/collection_log.py`|
| `python_app/classifier_app.py`        | `pipeline/02_annotation/app.py`           |
| `python_app/classifier/classifier.py` | `pipeline/02_annotation/classifier.py`    |
| `python_app/classifier/dashboard.py`  | `pipeline/03_eda/dashboard.py`            |
| `python_app/requirements.txt`         | `requirements.txt` (raiz)                 |
| `json_data/search_tems.json`          | `config/search_terms_monthly.json`        |
| `json_data/all_search.json`           | `config/search_terms_historical.json`     |
| `init-scripts/`                       | `infra/`                                  |
| `QUERIES`                             | `queries/analysis.sql`                    |

---

## 6. Decisões de design — não alterar sem discussão

### 6.1 BaseExtractor (01_extraction/base_extractor.py)

Classe abstrata que permite adicionar novas fontes de dados (ex: web scraping)
sem modificar o código existente. Toda nova fonte deve implementar esta interface:

```python
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    @abstractmethod
    def fetch(self, user_id: str, from_dt: str, to_dt: str) -> list[dict]:
        """Retorna lista de tweets/posts como dicionários padronizados."""
        ...
```

Implementação atual: `TwitterAPIExtractor(BaseExtractor)` em `twitter_api.py`.

### 6.2 shared/ para código transversal

`database.py` e `models.py` são usados por múltiplas etapas (01, 02, 04, 05).
Centralizá-los em `shared/` elimina imports circulares e duplicação de código.

### 6.3 Campo `classificator` no banco

A tabela `tweets_classification` usa o campo `classificator` para diferenciar
a origem de cada anotação:
- `"Humano"` → anotação via interface Streamlit (02_annotation)
- `"FinBERT-PT-BR"` → inferência do modelo (04_processing)

Isso permite comparar as duas fontes diretamente para avaliação (05_evaluation).

### 6.4 Numeração explícita das etapas

Os prefixos `01_` a `06_` tornam a ordem do pipeline legível no próprio
sistema de arquivos. Manter essa convenção em qualquer novo módulo.

---

## 7. Banco de dados (PostgreSQL via Docker)

### Tabela `tweets`
| Coluna          | Tipo        | Descrição                        |
|-----------------|-------------|----------------------------------|
| id              | SERIAL      | Chave primária                   |
| tweet_id        | VARCHAR(50) | ID único no X                    |
| username        | VARCHAR(100)| Nome de usuário                  |
| note_tweet      | TEXT        | Texto completo (tweets longos)   |
| created_at      | TIMESTAMPTZ | Data/hora de publicação          |
| likes           | INTEGER     | Curtidas                         |
| hashtags        | JSONB       | Lista de hashtags                |
| tweet           | JSONB       | Objeto completo da API           |
| sentiment       | VARCHAR(8)  | positivo / negativo / neutro     |
| is_finance_news | INTEGER     | 1 = financeiro, 0 = não          |

### Tabela `tweets_classification`
| Coluna               | Tipo         | Descrição                          |
|----------------------|--------------|------------------------------------|
| id                   | SERIAL       | Chave primária                     |
| tweet_id             | INTEGER      | FK → tweets.id                     |
| sentiment            | VARCHAR(8)   | Sentimento classificado            |
| why_sentiment        | TEXT         | Justificativa                      |
| is_finance_news      | INTEGER      | Classificação financeiro/não       |
| why_is_finance_news  | TEXT         | Justificativa                      |
| classificator        | VARCHAR(100) | "Humano" ou "FinBERT-PT-BR"        |

### Tabela `collection_log`
| Coluna           | Tipo        | Descrição                               |
|------------------|-------------|-----------------------------------------|
| id               | SERIAL      | Chave primária                          |
| search_term      | JSONB       | {x_user_id, from_date_time, to_date_time}|
| tweets_collected | INTEGER     | Quantidade de tweets inseridos          |
| start_time       | TIMESTAMP   | Início da coleta                        |
| end_time         | TIMESTAMP   | Fim da coleta                           |
| status           | VARCHAR(20) | pending / completed / partially_completed|
| error_message    | TEXT        | Mensagem de erro (se houver)            |

---

## 8. Implementação das etapas ausentes

### 03_eda/dashboard.py (migração + expansão)
O `dashboard.py` atual analisa distribuição de sentimentos e classificação
financeiro/não-financeiro. Expandir para incluir análise do texto bruto
que vai informar o pré-processamento: comprimento dos tweets, frequência
de emojis, URLs, menções, hashtags e distribuição temporal.

### 04_preprocessing/text_cleaner.py
Remove ruídos típicos de tweets para preparar o texto para o FinBERT-PT-BR.
As regras de limpeza devem ser guiadas pelos padrões encontrados na EDA:

```python
import re
import emoji

def clean(text: str) -> str:
    text = re.sub(r'http\S+', '', text)            # URLs
    text = emoji.replace_emoji(text, replace='')    # emojis
    text = re.sub(r'@\w+', '', text)               # menções
    text = re.sub(r'#(\w+)', r'\1', text)          # hashtags → texto
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### 05_processing/sentiment_analyzer.py
- Carrega o modelo FinBERT-PT-BR
- Recebe tweets pré-processados da tabela `tweets`
- Grava resultados em `tweets_classification` com `classificator = "FinBERT-PT-BR"`
- Reutiliza `shared/database.py` para acesso ao banco

### 06_evaluation/metrics.py
- Compara linhas com `classificator = "Humano"` vs `classificator = "FinBERT-PT-BR"`
  para os mesmos `tweet_id`
- Calcula: acurácia, F1 por classe (macro e weighted), matriz de confusão
- Exporta relatório de avaliação

---

## 9. Comandos (Makefile)

```makefile
collect:
    python -m pipeline.01_extraction

annotate:
    streamlit run pipeline/02_annotation/app.py

eda:
    streamlit run pipeline/03_eda/dashboard.py

preprocess:
    python -m pipeline.04_preprocessing

process:
    python -m pipeline.05_processing

evaluate:
    python -m pipeline.06_evaluation

db-up:
    docker-compose up -d

db-down:
    docker-compose down
```

---

## 10. Fontes de dados

| Conta X       | ID             | Volume no dataset |
|---------------|----------------|-------------------|
| InfoMoney     | 59773459       | ~88,5%            |
| InvestingBrasil | (ver .env)   | ~11,5%            |

Dataset total: ~3.563 tweets (outubro/2025 a fevereiro/2026).
Coleta via `GET /2/users/{id}/tweets` (API X v2, Bearer Token).

---

## 11. Repositórios

- **TCC (LaTeX):** https://github.com/toninlopes/usp-tcc-mba-ia-bigdata
- **Pipeline:** https://github.com/toninlopes/usp-tcc-pipeline-mba-ia-bigdata

---

*Última atualização: abril/2026*