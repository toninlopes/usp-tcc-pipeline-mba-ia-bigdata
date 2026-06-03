from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from app.core.classification.bert.bert_timbau import BERTimbauAnalyzer
from app.core.fine_tuning.bert_timbau_fine_tuner import OUTPUT_DIR, RUNS_FILE
from app.core.classification.bert.finbert_ptbr import FinBertPTBRAnalyzer
from app.core.classification.lexicon.op_lexicon import OpLexiconAnalyzer
from app.core.classification.lexicon.senti_lex import SentiLexAnalyzer
from app.shared.db.classification import ClassificationRepository
from app.shared.db.dataset_split import DatasetSplitRepository
from app.shared.text_cleaner import (
    lemmatize,
    lowercase_normalization,
    remove_emojis,
    remove_hashtags,
    remove_mentions,
    remove_stopwords,
    remove_urls,
    replace_emojis_with_codes,
    replace_mentions,
    replace_urls,
    space_normalization,
    strip_boundary_punctuation,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
BEST_CONFIG_PATH = _PROJECT_ROOT / "models" / "best_preprocessing.json"

classification_repo = ClassificationRepository()
split_repo = DatasetSplitRepository()

_bertimbau_ready = (OUTPUT_DIR / "config.json").exists()

ALGORITHMS = {
    "FinBERT-PT-BR": {"cls": FinBertPTBRAnalyzer, "ready": True,              "note": None},
    "BERTimbau":     {"cls": BERTimbauAnalyzer,    "ready": _bertimbau_ready,  "note": None if _bertimbau_ready else "Requer fine-tuning prévio. Execute: python -m app.core.fine_tuning.bert_timbau_fine_tuner"},
    "SentiLex-PT":   {"cls": SentiLexAnalyzer,     "ready": True,              "note": None},
    "OpLexicon":     {"cls": OpLexiconAnalyzer,     "ready": True,              "note": None},
}

# (label, function, warning) — order defines the execution pipeline
PREPROCESSING_STEPS: List[tuple] = [
    ("Substituir URLs por [URL]",            replace_urls,              None),
    ("Remover URLs",                         remove_urls,               None),
    ("Remover emojis",                       remove_emojis,             None),
    ("Substituir emojis por códigos",        replace_emojis_with_codes, None),
    ("Substituir menções por [MENTION]",     replace_mentions,          None),
    ("Remover menções (@)",                  remove_mentions,           None),
    ("Remover hashtags (#)",                 remove_hashtags,           None),
    ("Remover pontuação de borda (.!?,)",    strip_boundary_punctuation, None),
    ("Normalizar espaços",                   space_normalization,       None),
    ("Normalizar caixa (preserva tickers)",  lowercase_normalization,   None),
    ("Remover stopwords",                    remove_stopwords,          None),
    ("Lematizar",                            lemmatize,                  "Lento: carrega spaCy pt_core_news_lg"),
]

_FN_TO_LABEL = {fn: label for label, fn, _ in PREPROCESSING_STEPS}


# ── Deployed model helpers ────────────────────────────────────────────────────

def _get_deployed_model_info() -> Optional[Dict]:
    """Returns the training_runs.json entry marked is_saved_model=True, or None."""
    if not RUNS_FILE.exists():
        return None
    try:
        with open(RUNS_FILE, encoding="utf-8") as f:
            runs: List[Dict] = json.load(f)
        return next((r for r in runs if r.get("is_saved_model")), None)
    except Exception:
        return None


def _get_best_run() -> Optional[Dict]:
    """Returns the training_runs.json entry with the highest best_val_f1_macro, or None."""
    if not RUNS_FILE.exists():
        return None
    try:
        with open(RUNS_FILE, encoding="utf-8") as f:
            runs: List[Dict] = json.load(f)
        return max(runs, key=lambda r: r["best_val_f1_macro"]) if runs else None
    except Exception:
        return None


# ── Best-config persistence ───────────────────────────────────────────────────

def load_best_config() -> dict:
    if not BEST_CONFIG_PATH.exists():
        return {}
    try:
        with open(BEST_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_best_config(config: dict) -> None:
    BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def maybe_save(algorithm: str, steps: List[Callable], concordance: float) -> bool:
    """Saves steps + concordance for algorithm if concordance improves. Returns True if saved."""
    config = load_best_config()
    if concordance > config.get(algorithm, {}).get("concordance", -1.0):
        config[algorithm] = {
            "concordance": concordance,
            "steps": [_FN_TO_LABEL[fn] for fn in steps],
        }
        save_best_config(config)
        return True
    return False


# ── Analyzer loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_analyzer(algorithm: str):
    return ALGORITHMS[algorithm]["cls"]()


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def apply_preprocessing(text: str, steps: List[Callable[[str], str]]) -> str:
    for fn in steps:
        text = fn(text)
    return text


# ── Classification run ────────────────────────────────────────────────────────

def run_classification(
    algorithm: str,
    preprocessing_steps: List[Callable[[str], str]],
) -> pd.DataFrame:
    with st.spinner("Carregando tweets do conjunto de teste..."):
        df = split_repo.query_by_split("test")

    if df.empty:
        st.warning("Nenhum tweet no conjunto de teste. Execute o split do dataset primeiro.")
        return pd.DataFrame()

    with st.spinner(f"Carregando modelo {algorithm}..."):
        analyzer = load_analyzer(algorithm)

    rows = []
    progress = st.progress(0, text="Iniciando classificação...")
    status = st.empty()
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        status.caption(f"Classificando tweet {i + 1} de {total} — ID {row.tweet_id}")
        preprocessed = apply_preprocessing(row.note_tweet, preprocessing_steps)
        label, score, *_ = analyzer.predict(preprocessed)
        score = round(score, 4)

        rows.append({
            "tweet_id":          row.tweet_id,
            "texto":             row.note_tweet[:120] + "…" if len(row.note_tweet) > 120 else row.note_tweet,
            "sentimento_humano": row.sentiment,
            "sentimento_modelo": label,
            "confiança":         score,
            "concordância":      "✅" if row.sentiment == label else "❌",
        })
        progress.progress((i + 1) / total, text=f"Tweet {i + 1} / {total}")

    status.empty()
    progress.empty()
    return pd.DataFrame(rows)


# ── Page layout ───────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Classificação", page_icon="🤖")
st.title("🤖 Classificação de Sentimento")
st.caption("Classifica os tweets do conjunto de teste (hold-out) com o modelo selecionado e compara com a anotação humana.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configurações")
    algorithm = st.selectbox(
        "Algoritmo",
        options=list(ALGORITHMS.keys()),
        format_func=lambda k: k if ALGORITHMS[k]["ready"] else f"{k} ⚠️",
        key="clf_algorithm",
    )
    if ALGORITHMS[algorithm]["note"]:
        st.warning(ALGORITHMS[algorithm]["note"])

    if algorithm == "BERTimbau" and _bertimbau_ready:
        deployed = _get_deployed_model_info()
        if deployed:
            cfg = deployed.get("config", {})
            ts = deployed["timestamp"].replace("T", " ")
            f1 = deployed["best_val_f1_macro"]
            st.caption(
                f"Implantado: **F1 {f1:.4f}**  \n"
                f"lr `{cfg.get('learning_rate'):.0e}` · "
                f"batch `{cfg.get('per_device_train_batch_size')}` · "
                f"max_len `{cfg.get('max_length', 256)}`  \n"
                f"Treinado em {ts}"
            )

    # When the algorithm changes, restore the saved best checkboxes before widgets render
    if st.session_state.get("_clf_prev_algorithm") != algorithm:
        saved_steps = set(load_best_config().get(algorithm, {}).get("steps", []))
        for label, _, _ in PREPROCESSING_STEPS:
            st.session_state[f"step_{label}"] = label in saved_steps
        st.session_state["_clf_prev_algorithm"] = algorithm

    st.divider()
    st.subheader("Pré-processamento")
    st.caption("Aplicado ao texto antes de enviar ao modelo, na ordem abaixo.")

    selected_steps: List[Callable[[str], str]] = []
    for label, fn, warning in PREPROCESSING_STEPS:
        if st.checkbox(label, key=f"step_{label}"):
            selected_steps.append(fn)
        if warning:
            st.caption(f"⚠️ {warning}")

    st.divider()
    run = st.button("Classificar", use_container_width=True)

# ── Best-config banner ────────────────────────────────────────────────────────

model_best = load_best_config().get(algorithm)
if model_best:
    best_steps = model_best.get("steps", [])
    best_concordance = model_best.get("concordance", 0.0)
    steps_text = (
        ", ".join(f"**{s}**" for s in best_steps)
        if best_steps
        else "*nenhum pré-processamento extra*"
    )
    st.info(
        f"Melhor configuração salva para **{algorithm}** "
        f"— concordância **{best_concordance:.1%}**: {steps_text}"
    )

# ── BERTimbau deployment status ───────────────────────────────────────────────

if algorithm == "BERTimbau":
    if not _bertimbau_ready:
        st.warning(
            "O BERTimbau ainda não foi treinado. "
            "Execute o fine-tuning na página 🧠 Fine-tuning BERTimbau antes de classificar."
        )
    else:
        deployed = _get_deployed_model_info()
        best = _get_best_run()
        if deployed and best and best["timestamp"] != deployed["timestamp"]:
            st.warning(
                f"O treinamento de **{best['timestamp'].replace('T', ' ')}** "
                f"(F1 **{best['best_val_f1_macro']:.4f}**) supera o modelo implantado "
                f"(F1 {deployed['best_val_f1_macro']:.4f}). "
                "Acesse a página 🧠 Fine-tuning BERTimbau e inicie um novo treinamento para atualizar."
            )

# ── Run classification ────────────────────────────────────────────────────────

if run:
    result_df = run_classification(algorithm, selected_steps)
    if not result_df.empty:
        st.session_state["clf_pending"] = {
            "df": result_df,
            "algorithm": algorithm,
            "steps": selected_steps,
        }

# ── Results ───────────────────────────────────────────────────────────────────

if "clf_save_success" in st.session_state:
    st.success(st.session_state.pop("clf_save_success"))

pending = st.session_state.get("clf_pending")

if pending is not None:
    result_df   = pending["df"]
    pending_algo = pending["algorithm"]
    concordance_rate = (result_df["concordância"] == "✅").mean()

    st.subheader(f"Resultado — {pending_algo}")
    col1, col2 = st.columns(2)
    col1.metric("Tweets classificados", len(result_df))
    col2.metric("Taxa de concordância", f"{concordance_rate:.1%}")
    st.dataframe(result_df, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("💾 Salvar no banco de dados", type="primary", use_container_width=True):
        classificator_val = ALGORITHMS[pending_algo]["cls"].classificator
        for _, row in result_df.iterrows():
            classification_repo.upsert_tweets_classification(
                tweet_id=int(row["tweet_id"]),
                is_finance_news=1,
                sentiment=row["sentimento_modelo"],
                classificator=classificator_val,
                score=float(row["confiança"]),
            )
        saved = maybe_save(pending_algo, pending["steps"], concordance_rate)
        if saved:
            step_names = [_FN_TO_LABEL[fn] for fn in pending["steps"]]
            detail = ", ".join(step_names) if step_names else "nenhum pré-processamento extra"
            st.session_state["clf_save_success"] = (
                f"Salvo! Novo melhor resultado para **{pending_algo}**: "
                f"concordância **{concordance_rate:.1%}** ({detail})."
            )
        else:
            st.session_state["clf_save_success"] = (
                f"{len(result_df)} classificações salvas no banco de dados."
            )
        del st.session_state["clf_pending"]
        st.rerun()
    if c2.button("🗑️ Descartar resultados", use_container_width=True):
        del st.session_state["clf_pending"]
        st.rerun()

else:
    classificator = ALGORITHMS[algorithm]["cls"].classificator
    classified_df = classification_repo.query_classified_test_tweets(classificator)

    if not classified_df.empty:
        agreement_rate = (classified_df["concordância"] == "✅").mean()
        col1, col2 = st.columns(2)
        col1.metric("Tweets classificados", len(classified_df))
        col2.metric("Taxa de concordância", f"{agreement_rate:.1%}")
        st.dataframe(classified_df, use_container_width=True)
    else:
        st.info("Nenhum tweet classificado para este modelo. Clique em 'Classificar' para iniciar.")
