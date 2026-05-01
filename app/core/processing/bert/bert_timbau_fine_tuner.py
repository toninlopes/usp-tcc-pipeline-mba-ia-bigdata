"""
Fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased) para
classificação de sentimento de tweets financeiros em português brasileiro.

Produz o modelo ajustado em models/bert-timbau-sentiment/, usado pelo
BERTimbauAnalyzer via processing dashboard.

Uso:
    python -m app.core.processing.bert_timbau_fine_tuner
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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.training_args import TrainingArguments

from app.shared.db_tweets import TweetsRepository
from app.shared.text_cleaner import (
    remove_hashtags,
    replace_emojis_with_codes,
    replace_mentions,
    replace_urls,
    space_normalization,
)

# ── Configuração ──────────────────────────────────────────────────────────────

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"

# app/core/processing/ → app/core/ → app/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = _PROJECT_ROOT / "models" / "bert-timbau-sentiment"

MAX_LENGTH = 128
RANDOM_STATE = 42

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


# ── Carga e split dos dados ───────────────────────────────────────────────────

def load_labeled_data() -> pd.DataFrame:
    """Carrega tweets financeiros rotulados pelo anotador humano."""
    repo = TweetsRepository()
    df = repo.query_all_tweets_with_human_classification()

    df = df[df["sentiment"].isin(LABEL_TO_ID)].copy()
    df["note_tweet_clean"] = df["note_tweet"].astype(str).apply(preprocess)
    df = df[df["note_tweet_clean"].str.strip() != ""].reset_index(drop=True)
    df["label_id"] = df["sentiment"].map(LABEL_TO_ID)
    return df


def stratified_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split 70/15/15 estratificado por classe de sentimento."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["label_id"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ── Rotina principal ──────────────────────────────────────────────────────────

def main() -> None:
    print("Carregando dados rotulados...")
    df = load_labeled_data()
    print(f"  Total: {len(df)} tweets")
    print("  Distribuição de classes:")
    print(df["sentiment"].value_counts().to_string())

    if len(df) < 300:
        raise RuntimeError(
            f"Apenas {len(df)} tweets rotulados — insuficiente para fine-tuning "
            f"estável. Rotule mais tweets via `make annotate` (mínimo ~300)."
        )

    train_df, val_df, test_df = stratified_split(df)
    print(f"\nSplits: treino={len(train_df)} | val={len(val_df)} | teste={len(test_df)}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(LABEL_TO_ID.values())),
        y=train_df["label_id"].to_numpy(dtype=np.int64),
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(
        "\nPesos por classe:",
        {k: round(float(w), 3) for k, w in zip(LABEL_TO_ID.keys(), weights)},
    )

    print(f"\nCarregando tokenizer {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode(texts: List[str]) -> Dict:
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = TweetDataset(encode(train_df["note_tweet_clean"].tolist()), train_df["label_id"].tolist())
    val_ds = TweetDataset(encode(val_df["note_tweet_clean"].tolist()), val_df["label_id"].tolist())
    test_ds = TweetDataset(encode(test_df["note_tweet_clean"].tolist()), test_df["label_id"].tolist())

    print(f"\nCarregando modelo base {BASE_MODEL}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=12,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
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

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nIniciando fine-tuning...")
    trainer.train()

    print("\nAvaliando no conjunto de teste (hold-out)...")
    test_preds = trainer.predict(test_ds)
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

    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_dir = OUTPUT_DIR / "eval"
    eval_dir.mkdir(exist_ok=True)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    with open(OUTPUT_DIR / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(ID_TO_LABEL, f, ensure_ascii=False, indent=2)

    with open(eval_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    cm_df.to_csv(eval_dir / "confusion_matrix.csv")

    train_df[["id", "tweet_id", "sentiment"]].to_csv(eval_dir / "train_split.csv", index=False)
    val_df[["id", "tweet_id", "sentiment"]].to_csv(eval_dir / "val_split.csv", index=False)
    test_df[["id", "tweet_id", "sentiment"]].to_csv(eval_dir / "test_split.csv", index=False)

    print(f"\nModelo salvo em: {OUTPUT_DIR}")
    print(f"Relatórios e splits salvos em: {eval_dir}")


if __name__ == "__main__":
    main()