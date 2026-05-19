"""Página de fine-tuning do BERTimbau.

Permite configurar hiperparâmetros, iniciar o treinamento e acompanhar
o progresso em tempo real por fold/época.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from transformers import TrainerCallback, TrainerControl, TrainerState

import app.core.processing.bert.training_state as _ts
from app.core.processing.bert.bert_timbau_fine_tuner import (
    BASE_MODEL,
    N_FOLDS,
    OUTPUT_DIR,
    prepare_df,
    train_fold,
)
from app.shared.db.dataset_split import DatasetSplitRepository

# _ts.state is a module-level dict in a separately cached module.
# It survives Streamlit's per-rerun re-execution of this page file,
# and is directly accessible from background training threads.

RUNS_FILE = OUTPUT_DIR / "training_runs.json"

_CONFIG_KEYS = [
    "num_train_epochs",
    "learning_rate",
    "per_device_train_batch_size",
    "warmup_ratio",
    "weight_decay",
    "early_stopping_patience",
]


# ── Run history helpers ───────────────────────────────────────────────────────

def _load_runs() -> List[Dict]:
    if not RUNS_FILE.exists():
        return []
    try:
        with open(RUNS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_run(config: Dict, fold_results: List[Dict]) -> None:
    runs = _load_runs()
    best = max(fold_results, key=lambda r: r["val_f1_macro"])
    saved_config = {k: config[k] for k in _CONFIG_KEYS if k in config}
    runs.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": saved_config,
        "best_fold": best["fold"],
        "best_val_f1_macro": best["val_f1_macro"],
        "fold_results": fold_results,
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_FILE, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)


def _configs_match(c1: Dict, c2: Dict) -> bool:
    return all(c1.get(k) == c2.get(k) for k in _CONFIG_KEYS)


def _find_duplicate(config: Dict) -> Optional[Dict]:
    return next((r for r in _load_runs() if _configs_match(r["config"], config)), None)


# ── Custom HuggingFace Trainer callback ───────────────────────────────────────

class _ProgressCallback(TrainerCallback):
    def __init__(self, fold: int) -> None:
        self._fold = fold

    def on_epoch_begin(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        epoch_num = int(state.epoch) + 1
        total = int(args.num_train_epochs)
        _ts.state["current_epoch"][self._fold] = (epoch_num, total)

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        epoch = round(float(logs.get("epoch", 0)), 1)
        entry = {
            "epoch": epoch,
            "train_loss": logs.get("loss"),
            "eval_loss": logs.get("eval_loss"),
            "f1_macro": logs.get("eval_f1_macro"),
            "f1_weighted": logs.get("eval_f1_weighted"),
            "accuracy": logs.get("eval_accuracy"),
        }
        if any(v is not None for k, v in entry.items() if k != "epoch"):
            _ts.state["fold_logs"].setdefault(self._fold, []).append(entry)


# ── Background training thread ────────────────────────────────────────────────

def _start_training(config: Dict) -> None:
    _ts.reset()
    _ts.state["running"] = True
    threading.Thread(target=_run_training, args=(config,), daemon=True).start()


def _run_training(config: Dict) -> None:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    try:
        split_repo = DatasetSplitRepository()
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        test_df = prepare_df(split_repo.query_by_split("test"))
        if len(test_df) == 0:
            _ts.state["error"] = "Hold-out vazio. Re-gere o split via dashboard."
            _ts.state["running"] = False
            return

        fold_results: List[Dict] = []

        for fold in range(1, N_FOLDS + 1):
            _ts.state["current_fold"] = fold
            train_df = prepare_df(split_repo.query_train_excluding_fold(fold))
            val_df = prepare_df(split_repo.query_by_fold(fold))

            if len(train_df) < 50 or len(val_df) < 10:
                _ts.state["error"] = (
                    f"Fold {fold} com dados insuficientes: "
                    f"treino={len(train_df)}, val={len(val_df)}. "
                    "Rotule mais tweets e re-gere o split."
                )
                _ts.state["running"] = False
                return

            val_f1, _ = train_fold(
                fold=fold,
                train_df=train_df,
                val_df=val_df,
                tokenizer=tokenizer,
                training_args_override=config,
                extra_callbacks=[_ProgressCallback(fold)],
            )
            fold_results.append({"fold": fold, "val_f1_macro": val_f1})
            _ts.state["fold_results"] = list(fold_results)

        _ts.state["done"] = True
        _ts.state["running"] = False
        _save_run(config, fold_results)

    except Exception as exc:
        _ts.state["error"] = str(exc)
        _ts.state["running"] = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_fold_logs(logs: List[Dict]) -> pd.DataFrame:
    """Merges per-entry trainer logs into one row per epoch."""
    by_epoch: Dict[float, Dict] = {}
    for entry in logs:
        e = entry["epoch"]
        row = by_epoch.setdefault(e, {"Época": e})
        if entry.get("train_loss") is not None:
            row["Train Loss"] = round(entry["train_loss"], 4)
        if entry.get("eval_loss") is not None:
            row["Eval Loss"] = round(entry["eval_loss"], 4)
        if entry.get("f1_macro") is not None:
            row["F1 Macro"] = round(entry["f1_macro"], 4)
        if entry.get("f1_weighted") is not None:
            row["F1 Weighted"] = round(entry["f1_weighted"], 4)
        if entry.get("accuracy") is not None:
            row["Accuracy"] = round(entry["accuracy"], 4)
    return pd.DataFrame(sorted(by_epoch.values(), key=lambda r: r["Época"]))


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Fine-tuning BERTimbau", page_icon="🧠")
st.title("🧠 Fine-tuning BERTimbau")
st.caption(
    f"Treina o `{BASE_MODEL}` com K-Fold estratificado ({N_FOLDS} folds) "
    "usando o particionamento gerado na página ✂️ Split do Dataset."
)

# ── Sidebar: configuration & controls ────────────────────────────────────────

with st.sidebar:
    st.header("Hiperparâmetros")

    num_epochs = st.slider("Épocas máximas", min_value=1, max_value=100, value=12)
    lr = st.selectbox(
        "Learning rate",
        options=[1e-5, 2e-5, 3e-5, 5e-5],
        index=1,
        format_func=lambda x: f"{x:.0e}",
    )
    batch_size = st.selectbox("Batch size (treino)", options=[8, 16, 32], index=1)
    warmup_ratio = st.slider(
        "Warmup ratio", min_value=0.0, max_value=0.3, value=0.1, step=0.05
    )
    weight_decay = st.number_input(
        "Weight decay", min_value=0.0, max_value=0.1, value=0.01, step=0.005,
        format="%.3f",
    )
    patience = st.slider("Early stopping (paciência)", min_value=1, max_value=5, value=2)

    st.divider()

    is_running = _ts.state["running"]
    is_done = _ts.state["done"]
    has_error = bool(_ts.state["error"])

    if is_running:
        st.info(f"Treinando fold {_ts.state['current_fold']} / {N_FOLDS}…")

    split_assigned = DatasetSplitRepository().is_assigned()
    start_disabled = is_running or not split_assigned

    if not split_assigned:
        st.warning("Nenhum split encontrado. Execute ✂️ Split do Dataset primeiro.")

    if st.button(
        "▶️ Iniciar Treinamento",
        use_container_width=True,
        type="primary",
        disabled=start_disabled,
    ):
        pending = {
            "num_train_epochs": num_epochs,
            "learning_rate": lr,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size * 2,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "early_stopping_patience": patience,
        }
        duplicate = _find_duplicate(pending)
        if duplicate:
            st.session_state._pending_config = pending
            st.session_state._duplicate_run = duplicate
        else:
            _start_training(pending)
            st.rerun()

    if is_done or has_error:
        if st.button("🔄 Reiniciar", use_container_width=True):
            _ts.reset()
            st.rerun()


# ── Duplicate config dialog ───────────────────────────────────────────────────

@st.dialog("Hiperparâmetros já utilizados")
def _duplicate_dialog(existing: Dict, config: Dict) -> None:
    ts = existing["timestamp"].replace("T", " ")
    f1 = existing["best_val_f1_macro"]
    best_fold = existing["best_fold"]
    st.warning(
        f"Estes hiperparâmetros já foram usados em **{ts}** e produziram "
        f"F1 Macro = **{f1:.4f}** (melhor fold: {best_fold})."
    )
    st.write("Deseja iniciar o treinamento mesmo assim?")
    c1, c2 = st.columns(2)
    if c1.button("Cancelar", use_container_width=True):
        del st.session_state._pending_config
        del st.session_state._duplicate_run
        st.rerun()
    if c2.button("Continuar mesmo assim", use_container_width=True, type="primary"):
        del st.session_state._pending_config
        del st.session_state._duplicate_run
        _start_training(config)
        st.rerun()


if "_pending_config" in st.session_state:
    _duplicate_dialog(st.session_state._duplicate_run, st.session_state._pending_config)


# ── Main area: progress & results ─────────────────────────────────────────────

@st.fragment(run_every=2)
def _show_progress() -> None:
    running = _ts.state["running"]
    done = _ts.state["done"]
    error = _ts.state["error"]
    current_fold = _ts.state["current_fold"]
    fold_logs = _ts.state["fold_logs"]
    fold_results = _ts.state["fold_results"]

    # ── Training status ───────────────────────────────────────────────────────

    if error:
        st.error(f"Erro durante o treinamento: {error}")
    elif not running and not done and current_fold == 0:
        st.info(
            "Configure os hiperparâmetros no painel lateral e clique em "
            "**▶️ Iniciar Treinamento** para começar."
        )
    else:
        # ── Overall progress bar ──────────────────────────────────────────────

        completed = len(fold_results)
        pct = completed / N_FOLDS if done else max(current_fold - 1, 0) / N_FOLDS

        if done:
            st.success(f"Treinamento concluído! Modelo salvo em `{OUTPUT_DIR}`")
            st.progress(1.0, text=f"Todos os {N_FOLDS} folds concluídos.")
        else:
            st.progress(pct, text=f"Fold {current_fold} / {N_FOLDS} em andamento…")

        st.divider()

        # ── Per-fold expandable sections ──────────────────────────────────────

        current_epoch = _ts.state["current_epoch"]

        for fold in range(1, N_FOLDS + 1):
            logs = fold_logs.get(fold, [])
            result = next((r for r in fold_results if r["fold"] == fold), None)
            epoch_info = current_epoch.get(fold)

            if result is not None:
                label = f"Fold {fold}  —  F1 Macro (val): {result['val_f1_macro']:.4f}"
                expanded = False
            elif fold == current_fold and running:
                epoch_label = (
                    f"época {epoch_info[0]}/{epoch_info[1]}" if epoch_info else "iniciando…"
                )
                label = f"Fold {fold}  —  treinando ({epoch_label})"
                expanded = True
            else:
                label = f"Fold {fold}"
                expanded = False

            with st.expander(label, expanded=expanded):
                if fold == current_fold and running and epoch_info:
                    epoch_num, total_epochs = epoch_info
                    st.progress(
                        epoch_num / total_epochs,
                        text=f"Época {epoch_num} / {total_epochs}",
                    )

                if logs:
                    df = _merge_fold_logs(logs)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                elif fold == current_fold and running:
                    st.caption("Aguardando primeiro log de época…")
                else:
                    st.caption("Fold ainda não iniciado.")

        # ── K-Fold summary (shown when done) ─────────────────────────────────

        if done and fold_results:
            st.divider()
            st.subheader("Resumo K-Fold")

            best = max(fold_results, key=lambda r: r["val_f1_macro"])
            summary_rows = []
            for r in fold_results:
                marker = " ◀ melhor" if r["fold"] == best["fold"] else ""
                summary_rows.append({
                    "Fold": f"Fold {r['fold']}{marker}",
                    "F1 Macro (val)": round(r["val_f1_macro"], 4),
                })

            st.dataframe(
                pd.DataFrame(summary_rows), use_container_width=True, hide_index=True
            )
            c1, c2 = st.columns(2)
            c1.metric("Melhor fold", f"Fold {best['fold']}")
            c2.metric("F1 Macro (val)", f"{best['val_f1_macro']:.4f}")
            st.caption(f"Modelo salvo em: `{OUTPUT_DIR}`")

    # ── Run history (always shown, refreshes with the fragment) ───────────────

    runs = _load_runs()
    st.divider()
    st.subheader("Histórico de Treinamentos")

    if not runs:
        st.info("Nenhum treinamento registrado ainda.")
        return

    rows = []
    for r in reversed(runs):
        cfg = r["config"]
        rows.append({
            "Data": r["timestamp"].replace("T", " "),
            "Épocas": cfg.get("num_train_epochs"),
            "LR": f"{cfg.get('learning_rate'):.0e}",
            "Batch": cfg.get("per_device_train_batch_size"),
            "Warmup": cfg.get("warmup_ratio"),
            "Weight Decay": cfg.get("weight_decay"),
            "Paciência": cfg.get("early_stopping_patience"),
            "Melhor Fold": r.get("best_fold"),
            "F1 Macro (val)": round(r.get("best_val_f1_macro", 0.0), 4),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


_show_progress()
