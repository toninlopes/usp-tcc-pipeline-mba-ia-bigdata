import streamlit as st
import pandas as pd
import plotly.express as px

from app.shared.db.tweets import TweetsRepository

tweet_repo = TweetsRepository()

USERNAME_MAP = {"59773459": "InfoMoney", "49292227": "InvestingBrasil"}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = tweet_repo.query_all_tweets()
    df["username"] = df["username"].map(USERNAME_MAP).fillna(df["username"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["ano_mes"] = df["created_at"].dt.to_period("M").astype(str)
    df["char_count"] = df["note_tweet"].apply(lambda x: len(x) if pd.notna(x) else 0)
    df["token_count"] = df["note_tweet"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    df["sentiment"] = df["sentiment"].fillna("")
    df["is_finance_tweet"] = df["is_finance_tweet"].fillna(-1)
    return df


st.set_page_config(page_title="Exploração dos Dados", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 80%; padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()

st.title("🔍 Exploração dos Dados Coletados")
st.write("---")

# ── 1. Dimensões do Dataset ───────────────────────────────────────────────────

st.header("1. Dimensões do Dataset")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total de tweets", len(df))
col2.metric("Veículos", df["username"].nunique())
col3.metric("Período", f"{df['ano_mes'].min()} a {df['ano_mes'].max()}")
col4.metric("Financeiros", int((df["is_finance_tweet"] == 1).sum()))
col5.metric("Não financeiros", int((df["is_finance_tweet"] == 0).sum()))

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Distribuição de Sentimentos")
    sentiment_counts = df[df["sentiment"] != ""].copy()
    sentiment_counts = sentiment_counts["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentimento", "Quantidade"]
    fig = px.bar(sentiment_counts, x="Sentimento", y="Quantidade", color="Sentimento")
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("Financeiro vs. Não Financeiro")
    finance_counts = df["is_finance_tweet"].value_counts().reset_index()
    finance_counts.columns = ["Categoria", "Quantidade"]
    finance_counts["Categoria"] = finance_counts["Categoria"].map(
        {1: "Financeiro", 0: "Não financeiro", -1: "Não classificado"}
    )
    fig2 = px.pie(finance_counts, values="Quantidade", names="Categoria")
    st.plotly_chart(fig2, use_container_width=True)

st.write("---")

# ── 2. Distribuição Temporal ──────────────────────────────────────────────────

st.header("2. Distribuição Temporal")
temporal = df.groupby(["ano_mes", "username"]).size().reset_index(name="quantidade")
fig = px.bar(temporal, x="ano_mes", y="quantidade", color="username", barmode="group",
             labels={"ano_mes": "Mês", "quantidade": "Tweets", "username": "Veículo"})
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# ── 3. Comprimento dos Textos ─────────────────────────────────────────────────

st.header("3. Comprimento dos Textos")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    fig = px.histogram(df, x="char_count", nbins=50, title="Distribuição por Caracteres",
                       labels={"char_count": "Caracteres"})
    st.plotly_chart(fig, use_container_width=True)
with chart_col2:
    fig = px.histogram(df, x="token_count", nbins=50, title="Distribuição por Tokens",
                       labels={"token_count": "Tokens"})
    st.plotly_chart(fig, use_container_width=True)

st.write("---")

# ── 4. Campos Ausentes ────────────────────────────────────────────────────────

st.header("4. Campos Ausentes")
missing = df.isnull().sum().reset_index()
missing.columns = ["Campo", "Ausentes"]
missing = missing[missing["Ausentes"] > 0]
if missing.empty:
    st.success("Nenhum campo ausente encontrado.")
else:
    st.dataframe(missing)
