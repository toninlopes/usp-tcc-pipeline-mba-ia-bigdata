import streamlit as st
import pandas as pd

from app.shared.db.dataset_split import DatasetSplitRepository

_repo = DatasetSplitRepository()


def _show_summary(counts: dict) -> None:
    total = counts.get("test", 0) + counts.get("train", 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total particionado", total)
    c2.metric("Hold-out (test)", counts.get("test", 0))
    c3.metric("Treino (train)", counts.get("train", 0))

    folds = counts.get("folds", {})
    if folds:
        st.subheader("Distribuição dos Folds")
        fold_df = pd.DataFrame(
            [{"Fold": f"Fold {f}", "Tweets": c} for f, c in sorted(folds.items())]
        )
        st.dataframe(fold_df, use_container_width=True, hide_index=True)


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Split do Dataset", page_icon="✂️")
st.title("✂️ Split do Dataset")
st.caption(
    "Divide os tweets com anotação humana em hold-out de teste (~15%) e "
    "folds estratificados de treino (K-Fold, 4 folds)."
)

is_assigned = _repo.is_assigned()

with st.sidebar:
    st.header("Configurações")
    if is_assigned:
        st.success("Split já atribuído.")
    else:
        st.info("Nenhum split atribuído ainda.")

    force = st.checkbox(
        "Re-gerar split (apaga o existente)",
        value=False,
        disabled=not is_assigned,
    )
    run = st.button(
        "Re-gerar Split" if (is_assigned and force) else "Gerar Split",
        use_container_width=True,
        type="primary",
        disabled=is_assigned and not force,
    )

if run:
    with st.status("Gerando particionamento do dataset...", expanded=True) as status:
        st.write("Carregando tweets com anotação humana...")
        try:
            counts = _repo.assign_split(force=force)
            st.write(
                f"Split estratificado calculado — "
                f"{counts['test']} teste, {counts['train']} treino."
            )
            st.write("Dados persistidos no banco.")
            status.update(label="Particionamento concluído!", state="complete")
        except ValueError as e:
            status.update(label="Erro ao gerar split", state="error")
            st.error(str(e))
            counts = {}
        except Exception as e:
            status.update(label="Erro inesperado", state="error")
            st.error(f"Erro inesperado: {e}")
            counts = {}

    if counts:
        st.divider()
        st.subheader("Resumo do Particionamento")
        _show_summary(counts)

elif is_assigned:
    st.subheader("Resumo do Particionamento Atual")
    summary = _repo._count_splits()
    if summary:
        _show_summary(summary)

else:
    st.info("Nenhum split atribuído. Use o painel lateral para gerar o particionamento.")
