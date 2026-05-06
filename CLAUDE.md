# CLAUDE.md — Pipeline de Análise de Sentimento

Este arquivo orienta o Claude Code sobre a arquitetura, convenções e decisões
de design do projeto. Leia antes de qualquer modificação.

---

## 1. Visão geral

Pipeline de coleta, anotação e análise de sentimento de publicações do
X/Twitter relacionadas ao mercado financeiro brasileiro.

**Modelos avaliados:**
- **BERT:** FinBERT-PT-BR (`lucas-leme/FinBERT-PT-BR`) — especializado para o domínio financeiro em PT-BR; BERTimbau (`neuralmind/bert-base-portuguese-cased`) — pré-treinado em português, com fine-tuning sobre o dataset anotado.
- **Léxico:** SentiLex-PT02 — léxico de sentimento para o português; OpLexicon v3.0 — léxico de opinião para o português brasileiro.

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
│   │   │   ├── base_analyzer.py  ← contrato público (ABC)
│   │   │   ├── bert/             ← modelos baseados em BERT
│   │   │   │   ├── bert_analyzer.py
│   │   │   │   ├── finbert_ptbr.py
│   │   │   │   ├── bert_timbau.py
│   │   │   │   └── bert_timbau_fine_tuner.py
│   │   │   └── lexicon/          ← modelos baseados em léxico
│   │   │       ├── lexicon_analyzer.py
│   │   │       ├── senti_lex.py
│   │   │       └── op_lexicon.py
│   │   └── evaluation/           ← métricas de avaliação
│   ├── dashboard/                ← UI Streamlit (sem lógica de negócio)
│   │   ├── app.py                ← entrypoint multi-página
│   │   └── pages/                ← uma página por etapa do pipeline
│   │       ├── annotation.py     ← anotação manual tweet a tweet
│   │       ├── eda.py            ← analytics e exploração dos dados
│   │       ├── exploration.py    ← exploração dos dados brutos
│   │       ├── preprocessing.py  ← impacto do pré-processamento
│   │       ├── dataset_split.py  ← particionamento train/test/fold
│   │       ├── processing.py     ← inferência dos modelos
│   │       └── evaluation.py     ← métricas de avaliação
│   └── shared/                   ← código transversal
│       ├── db/                   ← acesso ao banco de dados
│       │   ├── database.py       ← pool de conexões PostgreSQL (DatabaseManager)
│       │   ├── tweets.py         ← queries da tabela tweets
│       │   ├── classification.py ← queries da tabela tweets_classification
│       │   ├── collection_log.py ← queries da tabela collection_log
│       │   └── dataset_split.py  ← queries da tabela dataset_split
│       ├── schemas.py            ← dataclasses de parsing da API X v2
│       └── text_cleaner.py       ← funções de limpeza textual
│
├── config/                       ← termos de busca (JSON)
├── data/                         ← léxicos (SentiLex-PT02, OpLexicon v3.0)
├── infra/                        ← scripts de inicialização do banco
├── models/                       ← modelos fine-tuned (gerados localmente)
│   └── bert-timbau-sentiment/    ← produzido por bert_timbau_fine_tuner.py
├── queries/                      ← SQL utilitário
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
from app.shared.db.tweets import TweetsRepository
from app.core.processing.bert.finbert_ptbr import FinBertPTBRAnalyzer
from app.core.processing.lexicon.senti_lex import SentiLexAnalyzer

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

| Tabela                  | Repositório                | Módulo                          |
|-------------------------|----------------------------|---------------------------------|
| `tweets`                | `TweetsRepository`         | `app.shared.db.tweets`          |
| `tweets_classification` | `ClassificationRepository` | `app.shared.db.classification`  |
| `collection_log`        | `CollectionLogRepository`  | `app.shared.db.collection_log`  |
| `dataset_split`         | `DatasetSplitRepository`   | `app.shared.db.dataset_split`   |

---

## 4. Hierarquia de modelos de sentimento

```
BaseSentimentAnalyzer              (processing/base_analyzer.py)
├── BertSentimentAnalyzer          (processing/bert/bert_analyzer.py)
│   ├── FinBertPTBRAnalyzer        (processing/bert/finbert_ptbr.py)
│   └── BERTimbauAnalyzer          (processing/bert/bert_timbau.py)
└── LexiconSentimentAnalyzer       (processing/lexicon/lexicon_analyzer.py)
    ├── SentiLexAnalyzer           (processing/lexicon/senti_lex.py)
    └── OpLexiconAnalyzer          (processing/lexicon/op_lexicon.py)
```

Para adicionar um novo modelo BERT:

```python
# app/core/processing/bert/my_bert.py
class MyBertAnalyzer(BertSentimentAnalyzer):
    model_name = "org/my-model"
    classificator = "MyModel"

    def load_model(self): ...
    def preprocess(self, text: str) -> str: ...
    def run(self) -> pd.DataFrame: ...
```

Para adicionar um novo léxico:

```python
# app/core/processing/lexicon/my_lexicon.py
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
| `"BERTimbau"`     | Inferência BERTimbauAnalyzer           |
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

### Tabela `dataset_split`

| Coluna     | Tipo        | Descrição                                                      |
|------------|-------------|----------------------------------------------------------------|
| `id`       | SERIAL      | Chave primária                                                 |
| `tweet_id` | INTEGER     | FK → `tweets.id`                                               |
| `split`    | VARCHAR(10) | `'train'` ou `'test'`                                          |
| `fold`     | SMALLINT    | Fold do K-Fold (1–4) para `train`; `NULL` para `test`          |

Restrições: `tweet_id` único; fold obrigatoriamente `NULL` quando `split='test'` e entre
1 e 4 quando `split='train'`. Gerenciada por `DatasetSplitRepository`
(`app/shared/db/dataset_split.py`).

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
python -m pytest app/ -v                          # todos os testes
python -m pytest app/shared/ -v                   # apenas shared
python -m pytest app/core/ -v                     # apenas core
python -m pytest app/core/processing/bert/ -v     # apenas modelos BERT
python -m pytest app/core/processing/lexicon/ -v  # apenas modelos léxicos
```

Fixtures e helpers compartilhados ficam em `app/shared/conftest.py`.

---

## 9. Fine-tuning do BERTimbau

O `BERTimbauAnalyzer` requer um modelo treinado localmente antes de ser usado.

```bash
# Treinar o modelo (requer mínimo ~300 tweets anotados)
python -m app.core.processing.bert.bert_timbau_fine_tuner

# O modelo será salvo em:
# models/bert-timbau-sentiment/
```

O fine-tuner usa os folds do `DatasetSplitRepository` (K-Fold estratificado),
`WeightedTrainer` para lidar com desbalanceamento de classes e
`EarlyStoppingCallback` com paciência de 2 épocas.

---

## 10. Léxicos de sentimento

Os arquivos de léxico não estão no repositório e são baixados automaticamente
na primeira instanciação de cada analisador.

| Léxico         | Caminho esperado                               | Fonte                             |
|----------------|------------------------------------------------|-----------------------------------|
| SentiLex-PT02  | `data/sentilex/sentiLex-PT02.txt`              | Download automático ao instanciar |
| OpLexicon v3.0 | `data/lexicons/oplexicon_v3.0/lexico_v3.0.txt` | Download automático ao instanciar |

---

## 11. Fontes de dados

| Conta X         | ID         | Volume no dataset |
|-----------------|------------|-------------------|
| InfoMoney       | `59773459` | ~88,5%            |
| InvestingBrasil | `51150679` | ~11,5%            |

Dataset: ~3.563 tweets (outubro/2025 a fevereiro/2026).
Coleta via `GET /2/users/{id}/tweets` (API X v2, Bearer Token).

---

## 12. Repositórios

- **TCC (LaTeX):** https://github.com/toninlopes/usp-tcc-mba-ia-bigdata
- **Pipeline:** https://github.com/toninlopes/usp-tcc-pipeline-mba-ia-bigdata

---

*Última atualização: maio/2026*