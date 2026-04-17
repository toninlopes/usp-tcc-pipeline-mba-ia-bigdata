import streamlit as st
import pandas as pd
from shared.database import DatabaseManager
from enum import Enum


class SelectedButtons(Enum):
    IS_FINANCE = "is_finance_button_selected"
    NOT_FINANCE = "not_finance_button_selected"
    POSITIVE = "positive_button_selected"
    NEUTRAL = "neutral_button_selected"
    NEGATIVE = "negative_button_selected"


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

db_manager = DatabaseManager()


@st.cache_data
def load_data():
    # Substitua pelo caminho do seu arquivo (ex: pd.read_csv('seu_arquivo.csv'))
    result = db_manager.query_all_tweets_with_human_classification()
    df = pd.DataFrame(
        result,
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
    df["id"] = df["id"].astype("int")
    df["created_at"] = df["created_at"].dt.strftime("%d/%m/%Y %H:%M:%S")
    df["sentiment"] = df["sentiment"].astype("string")
    df["sentiment"].fillna(
        "", inplace=True
    )  # Substitui NaN por string vazia para exibição
    df["is_finance_tweet"] = df["is_finance_tweet"].astype("Int64")
    df["is_finance_tweet"].fillna(
        -1, inplace=True
    )  # Substitui NaN por -1 para exibição
    df["has_human_classification"] = df["has_human_classification"].astype("bool")

    return df


def load_classification_data(tweet_id: int):
    result = db_manager.query_tweets_classification_by_id(tweet_id)
    df = pd.DataFrame(
        result,
        columns=[
            "id",
            "tweet_id",
            "sentiment",
            "why_sentiment",
            "is_finance_news",
            "why_is_finance_news",
            "classificator",
            "score",
        ],
    )
    df = df.drop(columns=["id", "tweet_id"])

    df["is_finance_news"] = df["is_finance_news"].apply(
        lambda x: "Sim" if x == 1 else "Não" if x == 0 else ""
    )

    return df


# Inicializa o dataframe na sessão para não resetar ao interagir
if "df" not in st.session_state:
    st.session_state.df = load_data()


# Inicializa o índice do registro atual
if "index" not in st.session_state:
    matching_indices = st.session_state.df[
        st.session_state.df["has_human_classification"] == False
    ].index
    st.session_state.index = matching_indices[0] if not matching_indices.empty else 0

df = st.session_state.df
idx = st.session_state.index


def save_is_finance_news(id: int, is_finance: int):
    result = db_manager.update_is_finance_news(id, is_finance)
    if result:
        df.at[idx, "is_finance_tweet"] = is_finance
        st.toast("Processing complete!", duration=5, icon="✅")
        st.rerun()
    else:
        st.toast(
            "Failed to update the database. Please try again.", duration=10, icon="❌"
        )


def save_sentiment(id: int, sentiment: str):
    result = db_manager.update_sentiment(id, sentiment)
    if result:
        df.at[idx, "sentiment"] = sentiment
        st.toast("Processing complete!", duration=5, icon="✅")
        st.rerun()
    else:
        st.toast(
            "Failed to update the database. Please try again.", duration=10, icon="❌"
        )


def save_classification_reasons(
    tweet_id: int,
    is_finance_news: int,
    why_is_finance_news: str,
    sentiment: str,
    why_sentiment: str,
    classificator: str,
):
    result = db_manager.insert_tweets_classification(
        tweet_id=tweet_id,
        is_finance_news=is_finance_news,
        why_is_finance_news=why_is_finance_news,
        sentiment=sentiment,
        why_sentiment=why_sentiment,
        classificator=classificator,
    )

    if result:
        st.rerun()
        st.toast("Classification details saved!", duration=5, icon="✅")
    else:
        st.toast(
            "Failed to save classification details. Please try again.",
            duration=10,
            icon="❌",
        )


## 1. Configurações da página
st.set_page_config(page_title="Rotulador de Sentimentos", layout="centered")
st.title("🏷️ Classificador de Tweets")

result = db_manager.check_connection()
if result == 0:
    st.subheader("Classifique os tweets como Positivo, Neutro ou Negativo")
else:
    st.error(
        "Falha na conexão com o banco de dados. Verifique as configurações e tente novamente."
    )

st.write("---")

# 2. Exibição do Texto
st.text(df.loc[idx, "note_tweet"])
st.text(df.loc[idx, "created_at"])

# 3. Botões de Classificação
st.write("---")

print(st.session_state.df.loc[idx])
is_finance_tweet = df.loc[idx, "is_finance_tweet"]
current_id = df.loc[idx, "id"].item()  # type: ignore # Convertendo para tipo nativo do Python

is_finance_col, is_not_finance_col = st.columns(2)
with is_finance_col:
    if st.button(
        "💰 É um tweet financeiro?",
        width="stretch",
        key=(SelectedButtons.IS_FINANCE.value if is_finance_tweet == 1 else None),
    ):
        save_is_finance_news(current_id, 1)  # type: ignore


with is_not_finance_col:
    if st.button(
        "❌ Não é um tweet financeiro?",
        width="stretch",
        key=(SelectedButtons.NOT_FINANCE.value if is_finance_tweet == 0 else None),
    ):
        save_is_finance_news(current_id, 0)  # type: ignore

if is_finance_tweet == 1:
    current_sentiment = df.loc[idx, "sentiment"]
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "😊 Positivo",
            width="stretch",
            key=(
                SelectedButtons.POSITIVE.value
                if current_sentiment == "positivo"
                else None  # type: ignore
            ),
        ):
            save_sentiment(current_id, "positivo")  # type: ignore

    with col2:
        if st.button(
            "😐 Neutro",
            width="stretch",
            key=(
                SelectedButtons.NEUTRAL.value if current_sentiment == "neutro" else None
            ),
        ):
            save_sentiment(current_id, "neutro")  # type: ignore

    with col3:
        if st.button(
            "😡 Negativo",
            width="stretch",
            key=(
                SelectedButtons.NEGATIVE.value
                if current_sentiment == "negativo"
                else None
            ),
        ):
            save_sentiment(current_id, "negativo")  # type: ignore

# 4. Navegação
st.write("---")
nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

with nav_col1:
    if st.button("⬅️ Anterior", width="stretch") and idx > 0:
        st.session_state.index -= 1
        st.rerun()

with nav_col2:
    st.progress(
        (idx + 1) / len(df),
        text=f"Tweet {idx + 1} de {len(df)} ({((idx + 1) / len(df)) * 100:.2f}%)",
    )

with nav_col3:
    if st.button("Próximo ➡️", width="stretch") and idx < len(df) - 1:
        st.session_state.index += 1
        st.rerun()

# 5. Exibição de Classificação Atual
st.write("---")


@st.dialog("Justificar Classificação")
def justify_classification():
    why_is_finance_news = st.text_area("Por que é (ou não) um tweet financeiro?")
    why_sentiment = st.text_area("Por que esse sentimento foi escolhido?")

    if st.button("Salvar", width="stretch"):
        save_classification_reasons(
            tweet_id=current_id,  # type: ignore
            is_finance_news=df.loc[idx, "is_finance_tweet"],  # type: ignore
            why_is_finance_news=why_is_finance_news.strip(),
            sentiment=df.loc[idx, "sentiment"],  # type: ignore
            why_sentiment=why_sentiment.strip(),
            classificator="Humano",
        )


class_table_col1, class_table_col2, class_table_col3 = st.columns([1, 1, 1])
with class_table_col1:
    st.subheader("Classificação atual")

with class_table_col3:
    if is_finance_tweet is not None:
        if st.button("Justificar", width="stretch"):
            justify_classification()

classification_data = load_classification_data(current_id)  # type: ignore
if not classification_data.empty:
    data_table = pd.DataFrame(
        {
            "Classificador": classification_data["classificator"].tolist(),
            "Financeiro": classification_data["is_finance_news"].tolist(),
            "Justificativa Financeiro": classification_data[
                "why_is_finance_news"
            ].tolist(),
            "Sentimento": classification_data["sentiment"].tolist(),
            "Justificativa Sentimento": classification_data["why_sentiment"].tolist(),
        }
    )
    st.dataframe(
        data_table,
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
