import streamlit as st
import pandas as pd
import plotly.express as px

from app.shared.db.tweets import TweetsRepository

tweet_repo = TweetsRepository()


def load_data() -> pd.DataFrame:
    df = tweet_repo.query_all_tweets()
    df["created_at"] = df["created_at"].dt.strftime("%d/%m/%Y %H:%M:%S")
    df["sentiment"] = df["sentiment"].fillna("")
    df["is_finance_tweet"] = df["is_finance_tweet"].fillna(-1)
    return df


df = load_data()

st.set_page_config(layout="wide")
st.title("📊 Analytics")
st.write("---")

col1, col2, col3 = st.columns(3)
col1.metric("Total de tweets", len(df))
col2.metric("Financeiros", int((df["is_finance_tweet"] == 1).sum()))
col3.metric("Não financeiros", int((df["is_finance_tweet"] == 0).sum()))

st.write("---")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Distribuição de Sentimentos")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
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