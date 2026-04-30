# CLAUDE.md — Pipeline de Análise de Sentimento

Este arquivo orienta o Claude Code sobre a arquitetura, convenções e decisões
de design do projeto. Leia antes de qualquer modificação.

---

## 1. Visão geral

Pipeline de coleta, anotação e análise de sentimento de publicações do
X/Twitter relacionadas ao mercado financeiro brasileiro.

**Ferramenta central:** FinBERT-PT-BR (`lucas-leme/FinBERT-PT-BR`)
**Produto final:** índice de sentimento avaliado contra gold standard humano

---

## 2. Estrutura do projeto

```
project/
│
├── app/                          ← todo o código Python
│   ├── core/                     ← lógica de negócio (sem UI)
│   │   ├── extraction/           ← coleta via API X v2
│   │   ├── processing/           ← inferência de sentimento
│   │   └── evaluation/           ← métricas de avaliação
│   ├── dashboard/                ← UI Streamlit (sem lógica de negócio)
│   │   ├── app.py                ← entrypoint multi-página
│   │   └── pages/                ← uma página por etapa do pipeline
│   └── shared/                   ← código transversal
│       ├── database.py           ← pool de conexões PostgreSQL
│       ├── db_tweets.py          ← queries da tabela tweets
│       ├── db_classification.py  ← queries da tabela tweets_classification
│       ├── db_collection_log.py  ← queries da tabela collection_log
│       ├── schemas.py            ← dataclasses de parsing da API X v2
│       └── text_cleaner.py       ← funções de limpeza textual
│
├── config/                       ← termos de busca (JSON)
├── data/                         ← léxicos (SentiLex-PT02, OpLexicon v3.0)
├── infra/                        ← scripts de inicialização do banco
├── queries/                      ← SQL utilitário
├── tests/
│   └── fixtures/                 ← JSONs de exemplo para testes manuais
├── .env
├── docker-compose.yml
├── Makefile
├── pytest.ini
└── requirements.txt
```

---

## 3. Convenções obrigatórias

### 3.1 PYTHONPATH

O projeto usa `PYTHONPATH=.` (raiz do projeto). Todos os imports são absolutos
a partir de `app.*`:

```python
# correto
from app.shared.db_tweets import TweetsRepository
from app.core.processing.finbert_ptbr import FinBertPTBRAnalyzer

# errado — nunca usar sys.path.insert
```

### 3.2 Compatibilidade Python 3.9

Usar sempre `Optional[X]` em vez de `X | None` e `List`, `Dict`, `Tuple`
de `typing` em vez das formas built-in do 3.10+.

### 3.3 Separação core / dashboard

- `app/core/` contém lógica de negócio pura — sem imports de `streamlit`
- `app/dashboard/` contém código Streamlit — sem lógica de negócio
- `app/shared/` contém código usado por ambos

### 3.4 Repositórios de banco

Nunca instanciar `DatabaseManager` diretamente nos módulos de negócio.
Usar os repositórios específicos:

| Tabela                  | Repositório                |
|-------------------------|----------------------------|
| `tweets`                | `TweetsRepository`         |
| `tweets_classification` | `ClassificationRepository` |
| `collection_log`        | `CollectionLogRepository`  |

---

## 4. Hierarquia de modelos de sentimento

```
BaseSentimentAnalyzer          (app/core/processing/base_analyzer.py)
├── BertSentimentAnalyzer      (app/core/processing/bert_analyzer.py)
│   └── FinBertPTBRAnalyzer    (app/core/processing/finbert_ptbr.py)
└── LexiconSentimentAnalyzer   (app/core/processing/lexicon_analyzer.py)
    ├── SentiLexAnalyzer       (app/core/processing/senti_lex.py)
    └── OpLexiconAnalyzer      (app/core/processing/op_lexicon.py)
```

Para adicionar um novo modelo BERT:

```python
class MyBertAnalyzer(BertSentimentAnalyzer):
    model_name = "org/my-model"
    classificator = "MyModel"

    def load_model(self): ...
    def preprocess(self, text: str) -> str: ...
    def run(self) -> pd.DataFrame: ...
```

Para adicionar um novo léxico:

```python
class MyLexiconAnalyzer(LexiconSentimentAnalyzer):
    classificator = "MyLexicon"

    def load_model(self) -> Dict[str, int]: ...
    def preprocess(self, text: str) -> str: ...
    def predict(self, text: str) -> Tuple[str, float]: ...
```

---

## 5. Campo `classificator` no banco

A tabela `tweets_classification` diferencia a origem de cada anotação:

| Valor             | Origem                                 |
|-------------------|----------------------------------------|
| `"Humano"`        | Anotação via dashboard (annotation.py) |
| `"FinBERT-PT-BR"` | Inferência FinBertPTBRAnalyzer         |
| `"SentiLex-PT"`   | Inferência SentiLexAnalyzer            |
| `"OpLexicon"`     | Inferência OpLexiconAnalyzer           |

---

## 6. Banco de dados (PostgreSQL via Docker)

### Tabela `tweets`

| Coluna             | Tipo         | Descrição                          |
|--------------------|--------------|----------------------------------- |
| `id`               | SERIAL       | Chave primária                     |
| `tweet_id`         | VARCHAR(50)  | ID único no X                      |
| `username`         | VARCHAR(100) | ID do autor                        |
| `note_tweet`       | TEXT         | Texto completo                     |
| `created_at`       | TIMESTAMPTZ  | Data/hora de publicação            |
| `likes`            | INTEGER      | Curtidas                           |
| `hashtags`         | JSONB        | Lista de hashtags                  |
| `tweet`            | JSONB        | Objeto completo da API             |
| `sentiment`        | VARCHAR(8)   | `positivo` / `negativo` / `neutro` |
| `is_finance_tweet` | INTEGER      | `1` = financeiro, `0` = não        |

### Tabela `tweets_classification`

| Coluna                | Tipo         | Descrição               |
|-----------------------|--------------|-------------------------|
| `id`                  | SERIAL       | Chave primária          |
| `tweet_id`            | INTEGER      | FK → `tweets.id`        |
| `sentiment`           | VARCHAR(8)   | Sentimento classificado |
| `why_sentiment`       | TEXT         | Justificativa           |
| `is_finance_news`     | INTEGER      | Financeiro / não        |
| `why_is_finance_news` | TEXT         | Justificativa           |
| `classificator`       | VARCHAR(100) | Origem da classificação |
| `score`               | REAL         | Confiança do modelo     |

### Tabela `collection_log`

| Coluna             | Tipo        | Descrição                                       |
|--------------------|-------------|-------------------------------------------------|
| `id`               | SERIAL      | Chave primária                                  |
| `search_term`      | JSONB       | `{x_user_id, from_date_time, to_date_time}`     |
| `tweets_collected` | INTEGER     | Quantidade inserida                             |
| `start_time`       | TIMESTAMP   | Início da coleta                                |
| `end_time`         | TIMESTAMP   | Fim da coleta                                   |
| `status`           | VARCHAR(20) | `pending` / `completed` / `partially_completed` |
| `error_message`    | TEXT        | Mensagem de erro (se houver)                    |

---

## 7. Comandos

```makefile
make collect      # coleta tweets via API X v2
make annotate     # abre dashboard Streamlit (anotação manual)
make eda          # abre dashboard Streamlit (exploração de dados)
make preprocess   # abre dashboard Streamlit (impacto do pré-processamento)
make process      # abre dashboard Streamlit (inferência dos modelos)
make evaluate     # executa métricas de avaliação (CLI)
make db-up        # sobe PostgreSQL + pgAdmin via Docker
make db-down      # para os containers
```

---

## 8. Testes

Testes ficam co-localizados com o módulo que testam (`*_tests.py`).

```bash
python -m pytest app/ -v          # todos os testes
python -m pytest app/shared/ -v   # apenas shared
python -m pytest app/core/ -v     # apenas core
```

Fixtures e helpers compartilhados ficam em `app/shared/conftest.py`.

---

## 9. Léxicos de sentimento

Os arquivos de léxico não estão no repositório e precisam ser obtidos separadamente.

| Léxico         | Caminho esperado                               | Fonte                             |
|----------------|------------------------------------------------|-----------------------------------|
| SentiLex-PT02  | `data/sentilex/sentiLex-PT02.txt`              | Download automático ao instanciar |
| OpLexicon v3.0 | `data/lexicons/oplexicon_v3.0/lexico_v3.0.txt` | Download automático ao instanciar |

---

## 10. Fontes de dados

| Conta X         | ID         | Volume no dataset |
|-----------------|------------|-------------------|
| InfoMoney       | `59773459` | ~88,5%            |
| InvestingBrasil | `51150679` | ~11,5%            |

Dataset: ~3.563 tweets (outubro/2025 a fevereiro/2026).
Coleta via `GET /2/users/{id}/tweets` (API X v2, Bearer Token).

---

## 11. Repositórios

- **TCC (LaTeX):** https://github.com/toninlopes/usp-tcc-mba-ia-bigdata
- **Pipeline:** https://github.com/toninlopes/usp-tcc-pipeline-mba-ia-bigdata

---

*Última atualização: maio/2026*