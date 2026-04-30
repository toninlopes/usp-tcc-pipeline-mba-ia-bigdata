import streamlit as st
import pandas as pd
from transformers.pipelines import pipeline as hf_pipeline
from transformers.models.bert import BertForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from app.shared.db_tweets import TweetsRepository
from app.shared.text_cleaner import (
    replace_urls, remove_emojis, replace_mentions, remove_hashtags,
    space_normalization, lowercase_normalization, remove_stopwords, lematize,
)

tweet_repo = TweetsRepository()

MODEL_NAME = "lucas-leme/FinBERT-PT-BR"
LABEL_MAP = {"NEGATIVE": "negativo", "NEUTRAL": "neutro", "POSITIVE": "positivo"}

STEPS = [
    ("URL", replace_urls),
    ("Emojis", remove_emojis),
    ("Menções", replace_mentions),
    ("Hashtags", remove_hashtags),
    ("Espaços", space_normalization),
    ("Caixa baixa", lowercase_normalization),
    ("Stopwords", remove_stopwords),
    ("Lematização", lematize),
]


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return hf_pipeline("text-classification", model=model, tokenizer=tokenizer)


def load_tweets(limit: int) -> pd.DataFrame:
    df = tweet_repo.query_all_tweets_with_human_classification()
    return (
        df[df["is_finance_tweet"] == 1]
        .dropna(subset=["note_tweet"])
        .head(limit)
        .reset_index(drop=True)
    )


def classify(pipe, text: str):
    result = pipe(text[:512], batch_size=1)[0]
    return LABEL_MAP.get(result["label"], result["label"]), round(result["score"], 4)


def apply_steps_cumulatively(text: str):
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
        st.warning("Nenhum tweet financeiro encontrado.")
        return pd.DataFrame()

    with st.spinner("Carregando modelo FinBERT-PT-BR..."):
        pipe = load_model()

    rows = []
    progress = st.progress(0, text="Executando inferência...")
    for i, row in enumerate(df.itertuples(index=False)):
        text = row.note_tweet
        step_texts = apply_steps_cumulatively(text)
        entry = {
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

st.set_page_config(layout="wide", page_title="Impacto do Pré-processamento", page_icon="🔬")
st.title("🔬 Impacto do Pré-processamento na Análise de Sentimento")
st.caption(
    "Cada coluna mostra o resultado do FinBERT-PT-BR após aplicar as etapas de "
    "pré-processamento de forma **cumulativa** (URL → Emojis → Menções → … → Lematização)."
)

with st.sidebar:
    st.header("Configurações")
    n_tweets = st.slider("Número de tweets", min_value=1, max_value=50, value=10)
    st.caption("⚠️ A lematização (spaCy) é lenta. Prefira valores baixos para testes iniciais.")
    run = st.button("Executar análise", use_container_width=True)

if run:
    result_df = run_analysis(n_tweets)
    if not result_df.empty:
        st.dataframe(result_df, use_container_width=True)