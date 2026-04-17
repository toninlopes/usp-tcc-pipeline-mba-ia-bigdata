import sys
import os
from pathlib import Path

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
from transformers.pipelines import pipeline as hf_pipeline
from transformers.models.bert import BertForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from shared.database import DatabaseManager
from shared.text_cleaner import (
    replace_urls,
    remove_emojis,
    replace_mentions,
    remove_hashtags,
    space_normalization,
    lowercase_normalization,
    remove_stopwords,
    lematize,
)

# Cada etapa é aplicada de forma cumulativa sobre o texto anterior.
STEPS: list[tuple[str, object]] = [
    ("URL", replace_urls),
    ("Emojis", remove_emojis),
    ("Menções", replace_mentions),
    ("Hashtags", remove_hashtags),
    ("Espaços", space_normalization),
    ("Caixa baixa", lowercase_normalization),
    ("Stopwords", remove_stopwords),
    ("Lematização", lematize),
]

MODEL_NAME = "lucas-leme/FinBERT-PT-BR"
LABEL_MAP = {
    "NEGATIVE": "negativo",
    "NEUTRAL": "neutro",
    "POSITIVE": "positivo",
}

db_manager = DatabaseManager()


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return hf_pipeline("text-classification", model=model, tokenizer=tokenizer)


def load_tweets(limit: int) -> pd.DataFrame:
    results = db_manager.query_all_tweets_with_human_classification()
    df = pd.DataFrame(
        results,
        columns=[
            "id",
            "tweet_id",
            "username",
            "note_tweet",
            "created_at",
            "likes",
            "hashtags",
            "tweet",
            "sentiment",
            "is_finance_tweet",
            "has_human_classification",
        ],
    )
    df = (
        df[df["is_finance_tweet"] == 1]
        .dropna(subset=["note_tweet"])
        .head(limit)
        .reset_index(drop=True)
    )
    return df


def classify(pipe, text: str) -> tuple[str, float]:
    """Executa inferência com o FinBERT-PT-BR e retorna (label_pt, score)."""
    result = pipe(text[:512], batch_size=1)[0]
    label = LABEL_MAP.get(result["label"], result["label"])
    return label, round(result["score"], 4)


def apply_steps_cumulatively(text: str) -> list[tuple[str, str]]:
    """Retorna lista de (step_name, texto_acumulado) após cada etapa."""
    results = []
    current = text
    for name, fn in STEPS:
        current = fn(current)
        results.append((name, current))
    return results


def run_analysis(n_tweets: int) -> pd.DataFrame:
    with st.spinner("Carregando tweets do banco..."):
        df = load_tweets(n_tweets)

    if df.empty:
        st.warning("Nenhum tweet financeiro encontrado no banco.")
        return pd.DataFrame()

    with st.spinner("Carregando modelo FinBERT-PT-BR..."):
        pipe = load_model()

    step_names = [name for name, _ in STEPS]
    rows = []
    progress = st.progress(0, text="Executando inferência...")

    for i, row in enumerate(df.itertuples(index=False)):
        text = row.note_tweet
        step_texts = apply_steps_cumulatively(text)

        entry: dict = {
            "tweet_id": row.tweet_id,
            "texto_original": text[:100] + "…" if len(text) > 100 else text,
            "humano": row.sentiment or "—",
        }

        for step_name, step_text in step_texts:
            label, score = classify(pipe, step_text)
            entry[step_name] = f"{label} ({score:.2f})"

        rows.append(entry)
        progress.progress((i + 1) / len(df), text=f"Tweet {i + 1} / {len(df)}")

    progress.empty()
    return pd.DataFrame(rows)


# ── Layout ────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Impacto do Pré-processamento",
    page_icon="🔬",
)

st.title("Impacto do Pré-processamento na Análise de Sentimento")
st.caption(
    "Cada coluna mostra o resultado do FinBERT-PT-BR após aplicar as etapas de "
    "pré-processamento de forma **cumulativa** (URL → Emojis → Menções → … → Lematização)."
)

with st.sidebar:
    st.header("Configurações")
    n_tweets = st.slider("Número de tweets", min_value=1, max_value=50, value=10)
    st.caption(
        "⚠️ A lematização (spaCy) é lenta. "
        "Prefira valores baixos para testes iniciais."
    )
    run_btn = st.button("Analisar", type="primary", use_container_width=True)

    if "results_df" in st.session_state:
        clear_btn = st.button("Limpar resultados", use_container_width=True)
        if clear_btn:
            del st.session_state["results_df"]
            st.rerun()

if run_btn:
    result_df = run_analysis(n_tweets)
    if not result_df.empty:
        st.session_state["results_df"] = result_df

if "results_df" in st.session_state:
    result_df = st.session_state["results_df"]

    step_columns = [name for name, _ in STEPS]

    st.subheader(f"Resultados — {len(result_df)} tweets")

    st.dataframe(
        result_df,
        use_container_width=True,
        height=600,
        column_config={
            "tweet_id": st.column_config.TextColumn("Tweet ID", width="small"),
            "texto_original": st.column_config.TextColumn(
                "Texto original", width="large"
            ),
            "humano": st.column_config.TextColumn("Humano", width="small"),
            **{
                col: st.column_config.TextColumn(col, width="medium")
                for col in step_columns
            },
        },
    )

    st.divider()

    # ── Análise de variação de sentimento por etapa ───────────────────────────
    st.subheader("Variação de sentimento por etapa")
    st.caption(
        "Percentual de tweets em que o sentimento predito difere do passo anterior."
    )

    prev_col = None
    variation_data = []

    for col in step_columns:
        if prev_col is None:
            prev_col = col
            continue
        changed = (
            result_df[col].str.split(" ").str[0]
            != result_df[prev_col].str.split(" ").str[0]
        ).sum()
        pct = round(changed / len(result_df) * 100, 1)
        variation_data.append(
            {"Etapa": col, "Tweets alterados": changed, "% alterados": pct}
        )
        prev_col = col

    st.dataframe(
        pd.DataFrame(variation_data),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "Configure os parâmetros na barra lateral e clique em **Analisar** para iniciar."
    )
