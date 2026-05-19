"""
Fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased) para
classificação de sentimento de tweets financeiros em português brasileiro.

Usa o particionamento estratificado persistido em `dataset_split` pelo
DatasetSplitRepository: hold-out fixo para avaliação final e K-Fold para
treinamento/validação cruzada.

Produz o modelo do melhor fold em models/bert-timbau-sentiment/, usado pelo
BERTimbauAnalyzer via processing dashboard.

Uso:
    PYTHONPATH=. python -m app.core.processing.bert.bert_timbau_fine_tuner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments

from app.shared.db.dataset_split import DatasetSplitRepository
from app.shared.text_cleaner import (
    remove_hashtags,
    replace_emojis_with_codes,
    replace_mentions,
    replace_urls,
    space_normalization,
)

# ── Configuração ──────────────────────────────────────────────────────────────

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"

# app/core/processing/bert/ → app/core/processing/ → app/core/ → app/ → root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = _PROJECT_ROOT / "models" / "bert-timbau-sentiment"

MAX_LENGTH = 128
RANDOM_STATE = 42
N_FOLDS = 4  # must match DatasetSplitRepository._N_SPLITS

LABEL_TO_ID: Dict[str, int] = {"negativo": 0, "neutro": 1, "positivo": 2}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}


# ── Pré-processamento ─────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Limpeza textual para BERTimbau cased.

    Preserva capitalização para não destruir tickers (PETR4),
    siglas (COPOM) e nomes próprios.
    """
    text = replace_urls(text)
    text = replace_emojis_with_codes(text)
    text = replace_mentions(text)
    text = remove_hashtags(text)
    text = space_normalization(text)
    return text


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica pré-processamento e mapeamento de labels a um DataFrame do DatasetSplitRepository.

    Args:
        df: DataFrame com colunas tweet_id, note_tweet, sentiment, fold.

    Returns:
        DataFrame enriquecido com note_tweet_clean e label_id, sem linhas inválidas.
    """
    df = df.copy()
    df["note_tweet_clean"] = df["note_tweet"].astype(str).apply(preprocess)
    df = df[df["note_tweet_clean"].str.strip() != ""]
    df = df[df["sentiment"].isin(LABEL_TO_ID)]
    df["label_id"] = df["sentiment"].map(LABEL_TO_ID)
    return df.reset_index(drop=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    """PyTorch Dataset empacotando encodings tokenizados e labels inteiros."""

    def __init__(self, encodings: Dict, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# ── Trainer com pesos por classe ──────────────────────────────────────────────

class WeightedTrainer(Trainer):
    """Trainer que aplica pesos por classe na CrossEntropyLoss.

    Mitiga o desbalanceamento típico de datasets de sentimento financeiro,
    onde tweets neutros tendem a ser mais frequentes.
    """

    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Acurácia, F1 macro e F1 ponderado. F1 macro é a métrica de seleção."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }


# ── Treino por fold ───────────────────────────────────────────────────────────

def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: Any,
    training_args_override: Optional[Dict] = None,
    extra_callbacks: Optional[List] = None,
) -> Tuple[float, Path]:
    """Treina o modelo para um único fold e retorna (val_f1_macro, model_dir).

    Args:
        fold: Número do fold (1..N_FOLDS).
        train_df: Dados de treino preparados (com note_tweet_clean e label_id).
        val_df: Dados de validação preparados.
        tokenizer: Tokenizer pré-carregado.

    Returns:
        Tupla (val_f1_macro, caminho do modelo salvo).
    """
    def encode(texts: List[str]) -> Dict:
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = TweetDataset(
        encode(train_df["note_tweet_clean"].tolist()),
        train_df["label_id"].tolist(),
    )
    val_ds = TweetDataset(
        encode(val_df["note_tweet_clean"].tolist()),
        val_df["label_id"].tolist(),
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(LABEL_TO_ID.values())),
        y=train_df["label_id"].to_numpy(dtype=np.int64),
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    overrides = training_args_override or {}
    fold_dir = OUTPUT_DIR / "checkpoints" / f"fold_{fold}"
    args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=overrides.get("num_train_epochs", 12),
        per_device_train_batch_size=overrides.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=overrides.get("per_device_eval_batch_size", 32),
        learning_rate=overrides.get("learning_rate", 2e-5),
        weight_decay=overrides.get("weight_decay", 0.01),
        warmup_ratio=overrides.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_STATE,
        report_to="none",
    )

    patience = overrides.get("early_stopping_patience", 2)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()

    val_metrics = trainer.evaluate()
    val_f1 = float(val_metrics.get("eval_f1_macro", 0.0))

    model_dir = fold_dir / "best"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_f1, model_dir


# ── Rotina principal ──────────────────────────────────────────────────────────

def main() -> None:
    split_repo = DatasetSplitRepository()

    if not split_repo.is_assigned():
        raise RuntimeError(
            "Nenhum split encontrado no banco. Execute a página "
            "'✂️ Split do Dataset' no dashboard antes de iniciar o fine-tuning."
        )

    print(f"Carregando tokenizer {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("\nCarregando hold-out de teste...")
    test_df = prepare_df(split_repo.query_by_split("test"))
    print(f"  Hold-out: {len(test_df)} tweets")

    if len(test_df) == 0:
        raise RuntimeError("Hold-out vazio. Re-gere o split via dashboard.")

    # ── K-Fold training ───────────────────────────────────────────────────────

    fold_results: List[Dict] = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'=' * 52}")
        print(f"  Fold {fold} / {N_FOLDS}")
        print(f"{'=' * 52}")

        train_df = prepare_df(split_repo.query_train_excluding_fold(fold))
        val_df = prepare_df(split_repo.query_by_fold(fold))

        if len(train_df) < 50 or len(val_df) < 10:
            raise RuntimeError(
                f"Fold {fold} com dados insuficientes: "
                f"treino={len(train_df)}, val={len(val_df)}. "
                f"Rotule mais tweets e re-gere o split."
            )

        print(f"  Treino: {len(train_df)} tweets")
        print(f"  Validação: {len(val_df)} tweets")
        print("  Distribuição (treino):")
        print(train_df["sentiment"].value_counts().to_string())

        val_f1, model_dir = train_fold(fold, train_df, val_df, tokenizer)

        fold_results.append({"fold": fold, "val_f1_macro": val_f1, "model_dir": model_dir})
        print(f"\n  Fold {fold} — F1 macro (val): {val_f1:.4f}")

    # ── Seleção do melhor fold ────────────────────────────────────────────────

    print(f"\n{'=' * 52}")
    print("  Resumo K-Fold")
    print(f"{'=' * 52}")
    for r in fold_results:
        marker = " ◀ melhor" if r == max(fold_results, key=lambda x: x["val_f1_macro"]) else ""
        print(f"  Fold {r['fold']}: F1 macro val = {r['val_f1_macro']:.4f}{marker}")

    best = max(fold_results, key=lambda x: x["val_f1_macro"])
    print(f"\nMelhor fold: {best['fold']} (F1 macro val: {best['val_f1_macro']:.4f})")

    # ── Avaliação no hold-out ─────────────────────────────────────────────────

    print("\nAvaliando no conjunto de teste (hold-out)...")

    best_model = AutoModelForSequenceClassification.from_pretrained(str(best["model_dir"]))

    def encode(texts: List[str]) -> Dict:
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    test_ds = TweetDataset(
        encode(test_df["note_tweet_clean"].tolist()),
        test_df["label_id"].tolist(),
    )

    eval_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_eval_tmp"),
        report_to="none",
    )
    eval_trainer = Trainer(
        model=best_model,
        args=eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    test_preds = eval_trainer.predict(test_ds)
    if test_preds.label_ids is None:
        raise RuntimeError("Predições sem labels no conjunto de teste.")

    raw_predictions = test_preds.predictions
    if isinstance(raw_predictions, tuple):
        raw_predictions = raw_predictions[0]

    y_true = np.asarray(test_preds.label_ids)
    y_pred = np.argmax(raw_predictions, axis=-1)
    label_names = [ID_TO_LABEL[i] for i in sorted(ID_TO_LABEL)]

    report = classification_report(y_true, y_pred, target_names=label_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in label_names],
        columns=[f"pred_{n}" for n in label_names],
    )

    print("\nClassification Report (hold-out):")
    print(report)
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # ── Salvar modelo e artefatos ─────────────────────────────────────────────

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_dir = OUTPUT_DIR / "eval"
    eval_dir.mkdir(exist_ok=True)

    eval_trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(ID_TO_LABEL, f, ensure_ascii=False, indent=2)

    with open(eval_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Melhor fold: {best['fold']} (F1 macro val: {best['val_f1_macro']:.4f})\n\n")
        f.write(report)

    cm_df.to_csv(eval_dir / "confusion_matrix.csv")

    kfold_summary = pd.DataFrame(fold_results).drop(columns=["model_dir"])
    kfold_summary.to_csv(eval_dir / "kfold_summary.csv", index=False)

    print(f"\nModelo salvo em: {OUTPUT_DIR}")
    print(f"Relatórios salvos em: {eval_dir}")


if __name__ == "__main__":
    main()
