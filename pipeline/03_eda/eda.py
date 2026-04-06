import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd
import plotly.express as px
from shared.database import DatabaseManager

db_manager = DatabaseManager()


@st.cache_data
def load_data():
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
    USERNAME_MAP = {"59773459": "InfoMoney", "49292227": "InvestingBrasil"}
    df["username"] = df["username"].map(USERNAME_MAP).fillna(df["username"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["ano_mes"] = df["created_at"].dt.to_period("M").astype(str)
    df["char_count"] = df["note_tweet"].apply(lambda x: len(x) if pd.notna(x) else 0)
    df["token_count"] = df["note_tweet"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    return df


st.set_page_config(page_title="Exploração dos Dados", layout="wide")

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

df = load_data()

st.title("Exploração dos Dados Coletados")
st.write("---")


# ── Seção 1: Dimensões do Dataset ───────────────────────────────────────────
st.header("1. Dimensões do Conjunto de Dados")

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Total de tweets", f"{len(df):,}")
col_m2.metric("Veículos monitorados", df["username"].nunique())
col_m3.metric(
    "Período coberto",
    f"{df['ano_mes'].min()} → {df['ano_mes'].max()}",
)

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tweets por Veículo")
    by_username = (
        df.groupby("username")
        .size()
        .reset_index(name="tweets")
        .sort_values("tweets", ascending=False)
    )
    fig_user = px.bar(
        by_username,
        x="username",
        y="tweets",
        labels={"username": "Veículo", "tweets": "Tweets"},
        color_discrete_sequence=["#3498db"],
    )
    fig_user.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_user, use_container_width=True)

with col2:
    st.subheader("Distribuição Temporal (por mês)")
    by_period = (
        df.groupby("ano_mes").size().reset_index(name="tweets").sort_values("ano_mes")
    )
    fig_period = px.bar(
        by_period,
        x="ano_mes",
        y="tweets",
        labels={"ano_mes": "Período", "tweets": "Tweets"},
        color_discrete_sequence=["#2ecc71"],
    )
    fig_period.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_period, use_container_width=True)

st.write("---")


# ── Seção 2: Completude dos Campos ──────────────────────────────────────────
st.header("2. Completude dos Campos")

DISPLAY_COLS = ["tweet_id", "username", "note_tweet", "created_at", "likes", "hashtags"]

missing = pd.DataFrame(
    {
        "campo": DISPLAY_COLS,
        "ausentes": [df[c].isnull().sum() for c in DISPLAY_COLS],
        "total": len(df),
    }
)
missing["percentual"] = (missing["ausentes"] / missing["total"] * 100).round(2)

col3, col4 = st.columns([2, 1])

with col3:
    fig_missing = px.bar(
        missing,
        x="campo",
        y="percentual",
        labels={"campo": "Campo", "percentual": "Valores ausentes (%)"},
        color="percentual",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 100],
        text="percentual",
    )
    fig_missing.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_missing.update_layout(coloraxis_showscale=False, yaxis_range=[0, 115])
    st.plotly_chart(fig_missing, use_container_width=True)

with col4:
    st.subheader("Resumo por Campo")
    display_missing = missing[["campo", "ausentes", "percentual"]].copy()
    display_missing.columns = ["Campo", "Ausentes", "% Ausente"]
    display_missing = display_missing.sort_values("Ausentes", ascending=False)
    st.dataframe(display_missing, use_container_width=True, hide_index=True)

    st.write("")
    note_tweet_missing = int(df["note_tweet"].isnull().sum())
    if note_tweet_missing > 0:
        st.warning(
            f"**{note_tweet_missing}** tweet(s) sem texto (`note_tweet` ausente). "
            "Esses registros devem ser excluídos antes da análise de conteúdo."
        )
    else:
        st.success("Campo `note_tweet` sem valores ausentes.")

st.write("---")


# ── Seção 3: Distribuição do Comprimento dos Textos ─────────────────────────
st.header("3. Distribuição do Comprimento dos Textos")

df_text = df[df["note_tweet"].notna()].copy()

LIMITE_CURTO_CHARS = 50
LIMITE_LONGO_CHARS = 1_000
LIMITE_CURTO_TOKENS = 5
LIMITE_LONGO_TOKENS = 150

col5, col6 = st.columns(2)

with col5:
    st.subheader("Por Número de Caracteres")
    fig_chars = px.histogram(
        df_text,
        x="char_count",
        nbins=60,
        labels={"char_count": "Caracteres", "count": "Frequência"},
        color_discrete_sequence=["#9b59b6"],
    )
    fig_chars.add_vline(
        x=LIMITE_CURTO_CHARS,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Curto (<{LIMITE_CURTO_CHARS})",
        annotation_position="top right",
    )
    fig_chars.add_vline(
        x=LIMITE_LONGO_CHARS,
        line_dash="dash",
        line_color="#e67e22",
        annotation_text=f"Longo (>{LIMITE_LONGO_CHARS})",
        annotation_position="top left",
    )
    st.plotly_chart(fig_chars, use_container_width=True)

with col6:
    st.subheader("Por Número de Tokens (palavras)")
    fig_tokens = px.histogram(
        df_text,
        x="token_count",
        nbins=60,
        labels={"token_count": "Tokens", "count": "Frequência"},
        color_discrete_sequence=["#1abc9c"],
    )
    fig_tokens.add_vline(
        x=LIMITE_CURTO_TOKENS,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Curto (<{LIMITE_CURTO_TOKENS})",
        annotation_position="top right",
    )
    fig_tokens.add_vline(
        x=LIMITE_LONGO_TOKENS,
        line_dash="dash",
        line_color="#e67e22",
        annotation_text=f"Longo (>{LIMITE_LONGO_TOKENS})",
        annotation_position="top left",
    )
    st.plotly_chart(fig_tokens, use_container_width=True)

# Estatísticas descritivas
col7, col8 = st.columns(2)

RENAME_STATS = {
    "count": "contagem",
    "mean": "média",
    "std": "desvio padrão",
    "min": "mínimo",
    "25%": "1º quartil",
    "50%": "mediana",
    "75%": "3º quartil",
    "max": "máximo",
}

with col7:
    st.write("**Estatísticas — Caracteres**")
    st.dataframe(
        df_text["char_count"]
        .describe()
        .rename(RENAME_STATS)
        .round(1)
        .to_frame("valor"),
        use_container_width=True,
    )

with col8:
    st.write("**Estatísticas — Tokens**")
    st.dataframe(
        df_text["token_count"]
        .describe()
        .rename(RENAME_STATS)
        .round(1)
        .to_frame("valor"),
        use_container_width=True,
    )

# Publicações atípicas
st.write("")
st.subheader("Publicações Atípicas")

curtos = df_text[df_text["char_count"] < LIMITE_CURTO_CHARS]
longos = df_text[df_text["char_count"] > LIMITE_LONGO_CHARS]

col9, col10 = st.columns(2)

with col9:
    st.metric(
        f"Textos curtos (< {LIMITE_CURTO_CHARS} caracteres)",
        f"{len(curtos):,}",
        delta=f"{len(curtos) / len(df_text) * 100:.1f}% do total",
        delta_color="inverse",
    )
    if not curtos.empty:
        st.dataframe(
            curtos[["tweet_id", "note_tweet", "char_count", "token_count"]].head(10),
            use_container_width=True,
            hide_index=True,
        )

with col10:
    st.metric(
        f"Textos longos (> {LIMITE_LONGO_CHARS} caracteres)",
        f"{len(longos):,}",
        delta=f"{len(longos) / len(df_text) * 100:.1f}% do total",
        delta_color="off",
    )
    if not longos.empty:
        st.dataframe(
            longos[["tweet_id", "note_tweet", "char_count", "token_count"]].head(10),
            use_container_width=True,
            hide_index=True,
        )
