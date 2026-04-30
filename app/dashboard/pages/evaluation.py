import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from app.core.evaluation.metrics import (
    evaluate, print_evaluation, LABELS, AVAILABLE_CLASSIFICATORS,
)


def plot_confusion_matrix(cm: np.ndarray, classificator: str):
    df_cm = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    fig = px.imshow(
        df_cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Matriz de Confusão — {classificator}",
        labels={"x": "Predito", "y": "Real"},
    )
    fig.update_layout(xaxis_title="Predito", yaxis_title="Real")
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Avaliação", page_icon="📈")
st.title("📈 Avaliação dos Modelos")
st.caption(
    "Compara as predições de cada modelo com o gold standard humano "
    "usando acurácia, F1-score e matriz de confusão."
)

with st.sidebar:
    st.header("Configurações")
    classificator = st.selectbox("Modelo a avaliar", options=AVAILABLE_CLASSIFICATORS)
    run = st.button("Avaliar", use_container_width=True)

if run:
    try:
        with st.spinner(f"Avaliando {classificator}..."):
            results = evaluate(classificator)

        col1, col2, col3 = st.columns(3)
        col1.metric("Amostras", results["n_samples"])
        col2.metric("Acurácia", f"{results['accuracy']:.4f}")
        col3.metric("F1 Macro", f"{results['f1_macro']:.4f}")

        st.write("---")
        chart_col, report_col = st.columns([1, 1])

        with chart_col:
            fig = plot_confusion_matrix(results["confusion_matrix"], classificator)
            st.plotly_chart(fig, use_container_width=True)

        with report_col:
            st.subheader("Relatório por Classe")
            st.code(results["report"], language=None)

    except ValueError as e:
        st.warning(str(e))