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


# ── Page config & CSS ─────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Filters ───────────────────────────────────────────────────────────────────

_FINANCE_OPTIONS = ["Todos", "Financeiro", "Não financeiro", "Não classificado"]
_SENTIMENT_OPTIONS = ["Todos", "Positivo", "Neutro", "Negativo"]

_FILTER_DEFAULTS = {
    "filter_finance": "Todos",
    "filter_sentiment": "Todos",
    "filter_no_human": False,
    "filter_unclassified": False,
}


def _init_filter_state() -> None:
    for key, val in _FILTER_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _on_filter_change() -> None:
    # Signal that we should move to the first tweet of the new filtered set.
    st.session_state.reset_to_first = True


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    finance = st.session_state.filter_finance
    if finance == "Financeiro":
        mask &= df["is_finance_tweet"] == 1
    elif finance == "Não financeiro":
        mask &= df["is_finance_tweet"] == 0
    elif finance == "Não classificado":
        mask &= df["is_finance_tweet"] == -1

    sentiment = st.session_state.filter_sentiment
    if sentiment != "Todos":
        mask &= df["sentiment"] == sentiment.lower()

    if st.session_state.filter_no_human:
        mask &= ~df["has_human_classification"]

    if st.session_state.filter_unclassified:
        mask &= df["is_finance_tweet"] == -1

    return df[mask]


# ── Session state ─────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = load_data()

_init_filter_state()

if "current_tweet_id" not in st.session_state:
    no_human = st.session_state.df[~st.session_state.df["has_human_classification"]]
    first = no_human.iloc[0] if not no_human.empty else st.session_state.df.iloc[0]
    st.session_state.current_tweet_id = int(first["id"])


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filtros")

    st.radio(
        "Financeiro",
        _FINANCE_OPTIONS,
        key="filter_finance",
        on_change=_on_filter_change,
    )
    st.divider()
    st.radio(
        "Sentimento",
        _SENTIMENT_OPTIONS,
        key="filter_sentiment",
        on_change=_on_filter_change,
    )
    st.divider()
    st.checkbox(
        "Sem classificação humana",
        key="filter_no_human",
        on_change=_on_filter_change,
    )
    st.checkbox(
        "Sem nenhuma classificação",
        key="filter_unclassified",
        on_change=_on_filter_change,
    )
    st.divider()
    if st.button("🔄 Recarregar dados", use_container_width=True):
        st.cache_data.clear()
        st.session_state.df = load_data()
        st.session_state.reset_to_first = True
        st.rerun()


# ── Filtered view & current tweet resolution ──────────────────────────────────

filtered_df = apply_filters(st.session_state.df)

# When a filter changes (or data reloads), jump to the first tweet in the new set.
if st.session_state.pop("reset_to_first", False):
    if not filtered_df.empty:
        st.session_state.current_tweet_id = int(filtered_df.iloc[0]["id"])

# Resolve current row from the full df — never affected by filter changes.
current_mask = st.session_state.df["id"] == st.session_state.current_tweet_id
current_rows = st.session_state.df[current_mask]
if current_rows.empty:
    current_rows = st.session_state.df.iloc[[0]]
row = current_rows.iloc[0]
actual_idx = row.name  # pandas label in full df, used for in-place updates

# Find where the current tweet sits within the filtered set (may be absent).
in_filter = not filtered_df.empty and (st.session_state.current_tweet_id in filtered_df["id"].values)
if in_filter:
    filter_pos = int(
        filtered_df.reset_index(drop=True)
        .index[filtered_df["id"].values == st.session_state.current_tweet_id][0]
    )
else:
    filter_pos = None


# ── Action handlers ───────────────────────────────────────────────────────────

def save_is_finance_news(actual_idx: int, tweet_id: int, is_finance: int) -> None:
    result = tweet_repo.update_is_finance_news(tweet_id, is_finance)
    if result:
        st.session_state.df.at[actual_idx, "is_finance_tweet"] = is_finance
        st.toast("Salvo!", duration=5, icon="✅")
        st.rerun()
    else:
        st.toast("Falha ao salvar. Tente novamente.", duration=10, icon="❌")


def save_sentiment(actual_idx: int, tweet_id: int, sentiment: str) -> None:
    result = tweet_repo.update_sentiment(tweet_id, sentiment)
    if result:
        st.session_state.df.at[actual_idx, "sentiment"] = sentiment
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
) -> None:
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

if tweet_repo.check_connection() != 0:
    st.error("Falha na conexão com o banco de dados.")

if filtered_df.empty and not in_filter:
    st.warning("Nenhum tweet encontrado com os filtros selecionados.")
    st.stop()

current_id = int(row["id"])
is_finance_tweet = row["is_finance_tweet"]
current_sentiment = row["sentiment"]

if not in_filter:
    st.info("Este tweet não está nos filtros ativos. Navegue para ver tweets do filtro.")

st.write("---")
st.text(row["note_tweet"])
st.text(row["created_at"])

st.write("---")
fin_col1, fin_col2 = st.columns(2)
with fin_col1:
    if st.button(
        "✅ É financeiro",
        key=SelectedButtons.IS_FINANCE.value if is_finance_tweet == 1 else "btn_is_finance",
        use_container_width=True,
    ):
        save_is_finance_news(actual_idx, current_id, 1)
with fin_col2:
    if st.button(
        "❌ Não é financeiro",
        key=SelectedButtons.NOT_FINANCE.value if is_finance_tweet == 0 else "btn_not_finance",
        use_container_width=True,
    ):
        save_is_finance_news(actual_idx, current_id, 0)

if is_finance_tweet == 1:
    st.write("---")
    sent_col1, sent_col2, sent_col3 = st.columns(3)
    with sent_col1:
        if st.button(
            "📈 Positivo",
            key=SelectedButtons.POSITIVE.value if current_sentiment == "positivo" else "btn_positive",
            use_container_width=True,
        ):
            save_sentiment(actual_idx, current_id, "positivo")
    with sent_col2:
        if st.button(
            "➡️ Neutro",
            key=SelectedButtons.NEUTRAL.value if current_sentiment == "neutro" else "btn_neutral",
            use_container_width=True,
        ):
            save_sentiment(actual_idx, current_id, "neutro")
    with sent_col3:
        if st.button(
            "📉 Negativo",
            key=SelectedButtons.NEGATIVE.value if current_sentiment == "negativo" else "btn_negative",
            use_container_width=True,
        ):
            save_sentiment(actual_idx, current_id, "negativo")

st.write("---")
nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

with nav_col1:
    prev_disabled = filtered_df.empty or (in_filter and filter_pos == 0)
    if st.button("⬅️ Anterior", disabled=prev_disabled, use_container_width=True):
        if in_filter and filter_pos > 0:
            st.session_state.current_tweet_id = int(filtered_df.iloc[filter_pos - 1]["id"])
        elif not in_filter:
            st.session_state.current_tweet_id = int(filtered_df.iloc[-1]["id"])
        st.rerun()

with nav_col2:
    if in_filter and not filtered_df.empty:
        pct = (filter_pos + 1) / len(filtered_df)
        label = f"Tweet {filter_pos + 1} de {len(filtered_df)} ({pct * 100:.2f}%)"
        st.progress(pct, text=label)
    elif not filtered_df.empty:
        st.progress(0.0, text=f"Tweet fora do filtro — {len(filtered_df)} no filtro")
    else:
        st.progress(0.0, text="Nenhum tweet no filtro")

with nav_col3:
    next_disabled = filtered_df.empty or (in_filter and filter_pos == len(filtered_df) - 1)
    if st.button("Próximo ➡️", disabled=next_disabled, use_container_width=True):
        if in_filter and filter_pos < len(filtered_df) - 1:
            st.session_state.current_tweet_id = int(filtered_df.iloc[filter_pos + 1]["id"])
        elif not in_filter:
            st.session_state.current_tweet_id = int(filtered_df.iloc[0]["id"])
        st.rerun()

st.write("---")


@st.dialog("Justificar Classificação")
def justify_classification() -> None:
    why_is_finance_news = st.text_area("Por que é (ou não) um tweet financeiro?")
    why_sentiment = st.text_area("Por que esse sentimento foi escolhido?")
    if st.button("Salvar", use_container_width=True):
        save_classification_reasons(
            tweet_id=current_id,
            is_finance_news=int(is_finance_tweet) if is_finance_tweet != -1 else 0,
            why_is_finance_news=why_is_finance_news.strip(),
            sentiment=str(current_sentiment),
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
