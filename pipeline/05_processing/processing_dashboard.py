from __future__ import annotations

import sys
import os
import types
import importlib.util
from pathlib import Path

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
from shared.database import DatabaseManager

# ── Load analyzer classes (relative imports require a synthetic package) ──────

_HERE = Path(__file__).resolve().parent
_PKG = "_pipeline_05_processing"

if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [str(_HERE)]
    _pkg.__package__ = _PKG
    sys.modules[_PKG] = _pkg


def _load_module(filename: str):
    name = f"{_PKG}.{Path(filename).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_load_module("base_model.py")
FinBERTAnalyzer = _load_module("fin_bert_timbau.py").FinBERTAnalyzer

try:
    BERTimbauAnalyzer = _load_module("bert_timbau.py").BERTimbauAnalyzer
    _bertimbau_available = True
except Exception:
    BERTimbauAnalyzer = None
    _bertimbau_available = False

# ── Constants ─────────────────────────────────────────────────────────────────

_FINE_TUNED_PATH = _HERE.parents[1] / "models" / "bert-timbau-sentiment"

ALGORITHMS: dict[str, dict] = {
    "FinBERT-PT-BR": {
        "cls": FinBERTAnalyzer,
        "ready": True,
        "note": None,
    },
}

if _bertimbau_available:
    ALGORITHMS["BERTimbau"] = {
        "cls": BERTimbauAnalyzer,
        "ready": _FINE_TUNED_PATH.exists(),
        "note": (
            None
            if _FINE_TUNED_PATH.exists()
            else "Modelo ainda não treinado — usando base sem fine-tuning."
        ),
    }

db_manager = DatabaseManager()


# ── Cached analyzer loader ────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_analyzer(algorithm: str):
    return ALGORITHMS[algorithm]["cls"]()


# ── Data helpers ──────────────────────────────────────────────────────────────


def fetch_tweets(limit: int) -> pd.DataFrame:
    rows = db_manager.query_all_tweets_with_human_classification()
    df = pd.DataFrame(
        rows,
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
    return (
        df[df["is_finance_tweet"] == 1]
        .dropna(subset=["note_tweet", "sentiment"])  # type: ignore[call-overload]
        .head(limit)
        .reset_index(drop=True)
    )


# ── Classification run ────────────────────────────────────────────────────────


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
        prediction = analyzer.predict_text(cleaned, batch_size=1)[0]
        model_label = analyzer.normalize_label(prediction["label"])
        score = round(prediction["score"], 4)

        if save_to_db:
            analyzer.db_manager.insert_tweets_classification(
                tweet_id=row.id,
                is_finance_news=1,
                why_is_finance_news="",
                sentiment=model_label,
                why_sentiment="",
                classificator=analyzer.classificator,
                score=score,
            )

        rows.append(
            {
                "tweet_id": row.tweet_id,
                "texto": (
                    row.note_tweet[:120] + "…"
                    if len(row.note_tweet) > 120
                    else row.note_tweet
                ),
                "sentimento_humano": row.sentiment,
                "sentimento_modelo": model_label,
                "confiança": score,
                "concordância": "✅" if row.sentiment == model_label else "❌",
            }
        )

        progress.progress((i + 1) / total, text=f"Tweet {i + 1} / {total}")

    status.empty()
    progress.empty()
    return pd.DataFrame(rows)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Classificação de Sentimento",
    page_icon="🤖",
)

st.title("Classificação de Sentimento")
st.caption(
    "Classifica tweets financeiros com o modelo selecionado e compara "
    "com a anotação humana existente."
)

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
        sample = fetch_tweets(limit=9999)
        available = len(sample)
    except Exception:
        available = 50

    n_tweets = st.slider(
        "Tweets a classificar",
        min_value=1,
        max_value=max(available, 1),
        value=min(10, max(available, 1)),
        help=f"{available} tweets com anotação humana disponíveis.",
    )
    st.caption(f"{available} tweets disponíveis.")

    save_to_db = st.checkbox("Salvar classificações no banco", value=False)

    st.divider()

    run_btn = st.button(
        "▶ Classificar",
        type="primary",
        use_container_width=True,
        disabled=(available == 0),
    )

    if "results_df" in st.session_state:
        if st.button("Limpar resultados", use_container_width=True):
            del st.session_state["results_df"]
            del st.session_state["results_algorithm"]
            st.rerun()

# ── Run ───────────────────────────────────────────────────────────────────────

if run_btn:
    result_df = run_classification(algorithm, n_tweets, save_to_db)
    if not result_df.empty:
        st.session_state["results_df"] = result_df
        st.session_state["results_algorithm"] = algorithm

# ── Results ───────────────────────────────────────────────────────────────────

if "results_df" in st.session_state:
    result_df: pd.DataFrame = st.session_state["results_df"]
    result_algorithm: str = st.session_state["results_algorithm"]

    total = len(result_df)
    agreements = (result_df["concordância"] == "✅").sum()
    dist = result_df["sentimento_modelo"].value_counts()

    st.subheader(f"Resultados — {result_algorithm} — {total} tweets")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total", total)
    col2.metric("Positivos", dist.get("positivo", 0))
    col3.metric("Neutros", dist.get("neutro", 0))
    col4.metric("Negativos", dist.get("negativo", 0))
    col5.metric("Concordância c/ humano", f"{agreements / total * 100:.1f}%")

    st.divider()

    st.subheader("Comparação tweet a tweet")
    st.dataframe(
        result_df,
        use_container_width=True,
        height=500,
        column_config={
            "tweet_id": st.column_config.TextColumn("Tweet ID", width="small"),
            "texto": st.column_config.TextColumn("Texto", width="large"),
            "sentimento_humano": st.column_config.TextColumn("Humano", width="small"),
            "sentimento_modelo": st.column_config.TextColumn("Modelo", width="small"),
            "confiança": st.column_config.NumberColumn(
                "Confiança", width="small", format="%.4f"
            ),
            "concordância": st.column_config.TextColumn("✓", width="small"),
        },
        hide_index=True,
    )

    disagreements = result_df[result_df["concordância"] == "❌"]
    if not disagreements.empty:
        with st.expander(
            f"Discordâncias ({len(disagreements)} tweets)", expanded=False
        ):
            st.dataframe(
                disagreements.drop(columns=["concordância"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tweet_id": st.column_config.TextColumn("Tweet ID", width="small"),
                    "texto": st.column_config.TextColumn("Texto", width="large"),
                    "sentimento_humano": st.column_config.TextColumn(
                        "Humano", width="small"
                    ),
                    "sentimento_modelo": st.column_config.TextColumn(
                        "Modelo", width="small"
                    ),
                    "confiança": st.column_config.NumberColumn(
                        "Confiança", width="small", format="%.4f"
                    ),
                },
            )

else:
    st.info(
        "Configure os parâmetros na barra lateral e clique em **▶ Classificar** para iniciar."
    )
