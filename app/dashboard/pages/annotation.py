import streamlit as st
import pandas as pd
from enum import Enum

from app.shared.db.tweets import TweetsRepository
from app.shared.db.classification import ClassificationRepository

tweet_repo = TweetsRepository()
classification_repo = ClassificationRepository()


class SelectedButtons(Enum):
    IS_FINANCE = "is_finance_button_selected"
    NOT_FINANCE = "not_finance_button_selected"
    POSITIVE = "positive_button_selected"
    NEUTRAL = "neutral_button_selected"
    NEGATIVE = "negative_button_selected"


st.set_page_config(page_title="Rotulador de Sentimentos", layout="centered")

for button in SelectedButtons:
    st.markdown(
        f"""
        <style>
        .st-key-{button.value} {{
            border: 1px solid #0f0;
            border-radius: 5px;
            color: white;
            z-index: 1000;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    .block-container {
        max-width: 80%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] td {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = tweet_repo.query_all_tweets_with_human_classification()
    df["id"] = df["id"].astype("int")
    df["created_at"] = df["created_at"].dt.strftime("%d/%m/%Y %H:%M:%S")
    df["sentiment"] = df["sentiment"].astype("string").fillna("")
    df["is_finance_tweet"] = df["is_finance_tweet"].astype("Int64").fillna(-1)
    df["has_human_classification"] = df["has_human_classification"].astype("bool")
    return df


def load_classification_data(tweet_id: int) -> pd.DataFrame:
    df = classification_repo.query_tweets_classification_by_id(tweet_id)
    if df.empty:
        return df
    df = df.drop(columns=["id", "tweet_id"])
    df["is_finance_news"] = df["is_finance_news"].apply(
        lambda x: "Sim" if x == 1 else "Não" if x == 0 else ""
    )
    return df


if "df" not in st.session_state:
    st.session_state.df = load_data()

if "index" not in st.session_state:
    matching_indices = st.session_state.df[
        st.session_state.df["has_human_classification"] == False
    ].index
    st.session_state.index = matching_indices[0] if not matching_indices.empty else 0

df = st.session_state.df
idx = st.session_state.index


def save_is_finance_news(id: int, is_finance: int):
    result = tweet_repo.update_is_finance_news(id, is_finance)
    if result:
        df.at[idx, "is_finance_tweet"] = is_finance
        st.toast("Salvo!", duration=5, icon="✅")
        st.rerun()
    else:
        st.toast("Falha ao salvar. Tente novamente.", duration=10, icon="❌")


def save_sentiment(id: int, sentiment: str):
    result = tweet_repo.update_sentiment(id, sentiment)
    if result:
        df.at[idx, "sentiment"] = sentiment
        st.toast("Salvo!", duration=5, icon="✅")
        st.rerun()
    else:
        st.toast("Falha ao salvar. Tente novamente.", duration=10, icon="❌")


def save_classification_reasons(
    tweet_id: int,
    is_finance_news: int,
    why_is_finance_news: str,
    sentiment: str,
    why_sentiment: str,
    classificator: str,
):
    result = classification_repo.insert_tweets_classification(
        tweet_id=tweet_id,
        is_finance_news=is_finance_news,
        why_is_finance_news=why_is_finance_news,
        sentiment=sentiment,
        why_sentiment=why_sentiment,
        classificator=classificator,
    )
    if result:
        st.rerun()
        st.toast("Justificativa salva!", duration=5, icon="✅")
    else:
        st.toast("Falha ao salvar. Tente novamente.", duration=10, icon="❌")


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🏷️ Classificador de Tweets")

if tweet_repo.check_connection() == 0:
    st.subheader("Classifique os tweets como Positivo, Neutro ou Negativo")
else:
    st.error("Falha na conexão com o banco de dados.")

st.write("---")

current_id = int(df.loc[idx, "id"])
is_finance_tweet = df.loc[idx, "is_finance_tweet"]
current_sentiment = df.loc[idx, "sentiment"]

st.text(df.loc[idx, "note_tweet"])
st.text(df.loc[idx, "created_at"])

st.write("---")
fin_col1, fin_col2 = st.columns(2)
with fin_col1:
    if st.button(
        "✅ É financeiro",
        key=SelectedButtons.IS_FINANCE.value if is_finance_tweet == 1 else "btn_is_finance",
        use_container_width=True,
    ):
        save_is_finance_news(current_id, 1)
with fin_col2:
    if st.button(
        "❌ Não é financeiro",
        key=SelectedButtons.NOT_FINANCE.value if is_finance_tweet == 0 else "btn_not_finance",
        use_container_width=True,
    ):
        save_is_finance_news(current_id, 0)

if is_finance_tweet == 1:
    st.write("---")
    sent_col1, sent_col2, sent_col3 = st.columns(3)
    with sent_col1:
        if st.button(
            "📈 Positivo",
            key=SelectedButtons.POSITIVE.value if current_sentiment == "positivo" else "btn_positive",
            use_container_width=True,
        ):
            save_sentiment(current_id, "positivo")
    with sent_col2:
        if st.button(
            "➡️ Neutro",
            key=SelectedButtons.NEUTRAL.value if current_sentiment == "neutro" else "btn_neutral",
            use_container_width=True,
        ):
            save_sentiment(current_id, "neutro")
    with sent_col3:
        if st.button(
            "📉 Negativo",
            key=SelectedButtons.NEGATIVE.value if current_sentiment == "negativo" else "btn_negative",
            use_container_width=True,
        ):
            save_sentiment(current_id, "negativo")

st.write("---")
nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
with nav_col1:
    if st.button("⬅️ Anterior", use_container_width=True) and idx > 0:
        st.session_state.index -= 1
        st.rerun()
with nav_col2:
    st.progress(
        (idx + 1) / len(df),
        text=f"Tweet {idx + 1} de {len(df)} ({((idx + 1) / len(df)) * 100:.2f}%)",
    )
with nav_col3:
    if st.button("Próximo ➡️", use_container_width=True) and idx < len(df) - 1:
        st.session_state.index += 1
        st.rerun()

st.write("---")

@st.dialog("Justificar Classificação")
def justify_classification():
    why_is_finance_news = st.text_area("Por que é (ou não) um tweet financeiro?")
    why_sentiment = st.text_area("Por que esse sentimento foi escolhido?")
    if st.button("Salvar", use_container_width=True):
        save_classification_reasons(
            tweet_id=current_id,
            is_finance_news=df.loc[idx, "is_finance_tweet"],
            why_is_finance_news=why_is_finance_news.strip(),
            sentiment=df.loc[idx, "sentiment"],
            why_sentiment=why_sentiment.strip(),
            classificator="Humano",
        )

col1, _, col3 = st.columns([1, 1, 1])
with col1:
    st.subheader("Classificação atual")
with col3:
    if is_finance_tweet is not None:
        if st.button("Justificar", use_container_width=True):
            justify_classification()

classification_data = load_classification_data(current_id)
if not classification_data.empty:
    st.dataframe(
        pd.DataFrame({
            "Classificador": classification_data["classificator"].tolist(),
            "Financeiro": classification_data["is_finance_news"].tolist(),
            "Justificativa Financeiro": classification_data["why_is_finance_news"].tolist(),
            "Sentimento": classification_data["sentiment"].tolist(),
            "Justificativa Sentimento": classification_data["why_sentiment"].tolist(),
        }),
        column_config={
            "Classificador": st.column_config.TextColumn(width="small"),
            "Financeiro": st.column_config.TextColumn(width="small"),
            "Justificativa Financeiro": st.column_config.TextColumn(width="medium"),
            "Sentimento": st.column_config.TextColumn(width="small"),
            "Justificativa Sentimento": st.column_config.TextColumn(width="large"),
        },
    )
else:
    st.info("Esse tweet ainda não possui justificativa para a classificação.")