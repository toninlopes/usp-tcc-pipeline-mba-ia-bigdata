import streamlit as st
import pandas as pd

from app.shared.db_tweets import TweetsRepository
from app.shared.db_classification import ClassificationRepository
from app.core.processing.finbert_ptbr import FinBertPTBRAnalyzer
from app.core.processing.senti_lex import SentiLexAnalyzer
from app.core.processing.op_lexicon import OpLexiconAnalyzer

tweet_repo = TweetsRepository()
classification_repo = ClassificationRepository()

ALGORITHMS = {
    "FinBERT-PT-BR": {"cls": FinBertPTBRAnalyzer, "ready": True, "note": None},
    "SentiLex-PT":   {"cls": SentiLexAnalyzer,    "ready": True, "note": None},
    "OpLexicon":     {"cls": OpLexiconAnalyzer,    "ready": True, "note": None},
}


@st.cache_resource(show_spinner=False)
def load_analyzer(algorithm: str):
    return ALGORITHMS[algorithm]["cls"]()


def fetch_tweets(limit: int) -> pd.DataFrame:
    df = tweet_repo.query_all_tweets_with_human_classification()
    return (
        df[(df["is_finance_tweet"] == 1) & (df["has_human_classification"] == True)]
        .dropna(subset=["note_tweet", "sentiment"])
        .head(limit)
        .reset_index(drop=True)
    )


def run_classification(algorithm: str, n_tweets: int, save_to_db: bool) -> pd.DataFrame:
    with st.spinner("Carregando tweets do banco..."):
        df = fetch_tweets(n_tweets)

    if df.empty:
        st.warning("Nenhum tweet financeiro com anotação humana encontrado.")
        return pd.DataFrame()

    with st.spinner(f"Carregando modelo {algorithm}..."):
        analyzer = load_analyzer(algorithm)

    rows = []
    progress = st.progress(0, text="Iniciando classificação...")
    status = st.empty()
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        status.caption(f"Classificando tweet {i + 1} de {total} — ID {row.tweet_id}")
        cleaned = analyzer.preprocess(row.note_tweet)
        label, score = analyzer.predict(cleaned)
        score = round(score, 4)

        if save_to_db:
            classification_repo.insert_tweets_classification(
                tweet_id=row.id,
                is_finance_news=1,
                why_is_finance_news="",
                sentiment=label,
                why_sentiment="",
                classificator=analyzer.classificator,
                score=score,
            )

        rows.append({
            "tweet_id": row.tweet_id,
            "texto": row.note_tweet[:120] + "…" if len(row.note_tweet) > 120 else row.note_tweet,
            "sentimento_humano": row.sentiment,
            "sentimento_modelo": label,
            "confiança": score,
            "concordância": "✅" if row.sentiment == label else "❌",
        })
        progress.progress((i + 1) / total, text=f"Tweet {i + 1} / {total}")

    status.empty()
    progress.empty()
    return pd.DataFrame(rows)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Classificação de Sentimento", page_icon="🤖")
st.title("🤖 Classificação de Sentimento")
st.caption("Classifica tweets financeiros com o modelo selecionado e compara com a anotação humana.")

with st.sidebar:
    st.header("Configurações")
    algorithm = st.selectbox(
        "Algoritmo",
        options=list(ALGORITHMS.keys()),
        format_func=lambda k: k if ALGORITHMS[k]["ready"] else f"{k} ⚠️",
    )
    if ALGORITHMS[algorithm]["note"]:
        st.warning(ALGORITHMS[algorithm]["note"])
    st.divider()
    try:
        available = len(fetch_tweets(limit=9999))
    except Exception:
        available = 50
    n_tweets = st.slider(
        "Tweets a classificar",
        min_value=1, max_value=max(available, 1),
        value=min(10, max(available, 1)),
        help=f"{available} tweets disponíveis.",
    )
    st.caption(f"{available} tweets disponíveis.")
    save_to_db = st.checkbox("Salvar no banco", value=False)
    run = st.button("Classificar", use_container_width=True)

if run:
    result_df = run_classification(algorithm, n_tweets, save_to_db)
    if not result_df.empty:
        agreement_rate = (result_df["concordância"] == "✅").mean()
        st.metric("Taxa de concordância", f"{agreement_rate:.1%}")
        st.dataframe(result_df, use_container_width=True)