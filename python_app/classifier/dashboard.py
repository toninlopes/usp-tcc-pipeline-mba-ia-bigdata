import streamlit as st
import pandas as pd
import plotly.express as px
from database import DatabaseManager


db_manager = DatabaseManager()


def load_data():
    # Substitua pelo caminho do seu arquivo (ex: pd.read_csv('seu_arquivo.csv'))
    result = db_manager.query_all_tweets()
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
        ],
    )
    df["created_at"] = df["created_at"].dt.strftime("%d/%m/%Y %H:%M:%S")
    df["sentiment"].fillna(
        "", inplace=True
    )  # Substitui NaN por string vazia para exibição
    # df["is_finance_tweet"] = df["is_finance_tweet"].astype("Int64")
    df["is_finance_tweet"].fillna(
        -1, inplace=True
    )  # Substitui NaN por -1 para exibição

    return df


df = load_data()

# 1. Set the page to wide mode
st.set_page_config(layout="wide")

# 2. Inject CSS to restrict the main container to 80%
st.markdown(
    """
    <style>
    .block-container {
        max-width: 80%;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Análise de Sentimentos dos Tweets Financeiros")

# Create two columns for the charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tweets de Notícias Financeiras")

    # Map 0/1 to labels for the chart
    relevance_counts = df["is_finance_tweet"].value_counts().reset_index()
    relevance_counts.columns = ["status", "count"]
    relevance_counts["label"] = relevance_counts["status"].map(
        {1: "Financeira", 0: "Não Financeira", -1: "Não Classificado"}
    )

    fig_relevance = px.pie(
        relevance_counts,
        values="count",
        names="label",
        color="label",
        color_discrete_map={
            "Financeira": "#2ecc71",
            "Não Financeira": "#e74c3c",
            "Não Classificado": "#f39c12",
        },
        hole=0.3,  # Optional: makes it a donut chart
    )
    st.plotly_chart(fig_relevance, use_container_width=True)

    st.write("---")
    st.subheader("Summary")
    st.text(f"Total de tweets financeiros: {df[df['is_finance_tweet'] == 1].shape[0]}")
    st.text(
        f"Total de tweets não financeiros: {df[df['is_finance_tweet'] == 0].shape[0]}"
    )
    st.text(
        f"Total de tweets não classificados: {df[df['is_finance_tweet'] == -1].shape[0]}"
    )

with col2:
    st.subheader("Distribuição de Sentimentos")

    # Filter for only relevant tweets if you want sentiment for market news only
    sentiment_df = df[df["is_finance_tweet"] == 1]

    sentiment_counts = sentiment_df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["label", "count"]
    sentiment_counts["label"] = sentiment_counts["label"].map(
        {
            "positivo": "Positivo",
            "negativo": "Negativo",
            "neutro": "Neutro",
        }
    )

    # Custom colors for sentiments
    color_map = {
        "Positivo": "#27ae60",
        "Negativo": "#c0392b",
        "Neutro": "#f1c40f",
    }

    fig_sentiment = px.pie(
        sentiment_counts,
        values="count",
        names="label",
        color="label",
        color_discrete_map=color_map,
        hole=0.3,
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    st.write("---")
    st.subheader("Summary")
    st.text(f"Total de tweets positivos: {df[df['sentiment'] == 'positivo'].shape[0]}")
    st.text(f"Total de tweets negativos: {df[df['sentiment'] == 'negativo'].shape[0]}")
    st.text(f"Total de tweets neutros: {df[df['sentiment'] == 'neutro'].shape[0]}")
