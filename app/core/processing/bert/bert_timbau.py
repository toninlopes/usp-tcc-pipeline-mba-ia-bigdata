from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline

from app.core.processing.bert.bert_analyzer import BertSentimentAnalyzer
from app.shared.text_cleaner import (
    replace_urls,
    replace_emojis_with_codes,
    replace_mentions,
    remove_hashtags,
    space_normalization,
)

# app/core/processing/ → app/core/ → app/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_FINE_TUNED_PATH = _PROJECT_ROOT / "models" / "bert-timbau-sentiment"


class BERTimbauAnalyzer(BertSentimentAnalyzer):
    """BERTimbau fine-tuned para sentimento financeiro em PT-BR.

    Requer modelo treinado em models/bert-timbau-sentiment/ produzido por
    bert_timbau_fine_tuner.py. Execute antes de instanciar:

        python -m app.core.processing.bert_timbau_fine_tuner
    """

    model_name = str(_FINE_TUNED_PATH)
    classificator = "BERTimbau"

    def __init__(self) -> None:
        if not _FINE_TUNED_PATH.exists():
            raise RuntimeError(
                f"Modelo fine-tuned não encontrado em {_FINE_TUNED_PATH}.\n"
                f"Execute: python -m app.core.processing.bert_timbau_fine_tuner"
            )
        super().__init__()
        self._model = self.load_model()

    def load_model(self):
        """Carrega o modelo fine-tuned BERTimbau do disco."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    def preprocess(self, text: str) -> str:
        """Limpeza sem lowercase — preserva capitalização de tickers e siglas.

        O fine-tuning foi feito com o modelo cased, então lowercase_normalization
        não deve ser aplicado aqui para manter consistência com o treino.
        """
        text = replace_urls(text)
        text = replace_emojis_with_codes(text)
        text = replace_mentions(text)
        text = remove_hashtags(text)
        text = space_normalization(text)
        return text

    def normalize_label(self, label: str) -> str:
        """O modelo fine-tuned emite labels em português via id2label.

        Preserva esses valores e mantém fallback para labels em inglês
        do HuggingFace por compatibilidade com a classe base.
        """
        if label in {"positivo", "negativo", "neutro"}:
            return label
        return super().normalize_label(label)

    def predict(self, text: str) -> Tuple[str, float]:
        """Executa inferência via pipeline HuggingFace.

        Sobrescreve BertSentimentAnalyzer.predict para usar batch_size=32
        e aplicar normalize_label antes de retornar.

        Returns:
            Tupla (label_pt, score).
        """
        result = self._model(text[:512], batch_size=32)[0]
        return self.normalize_label(result["label"]), result["score"]

    def run(self) -> pd.DataFrame:
        """Busca tweets com classificação humana, classifica e retorna resultados."""
        rows = self._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification()

        if rows.empty:
            print("[BERTimbau] Nenhum tweet encontrado para classificação.")
            return pd.DataFrame()

        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)
        rows["predicted_sentiment"] = rows["clear_tweets"].apply(
            lambda text: [{"label": lbl, "score": sc}
                          for lbl, sc in [self.predict(text)]]
        )
        return rows


if __name__ == "__main__":
    analyzer = BERTimbauAnalyzer()
    result = analyzer.run()
    if not result.empty:
        print(result[["tweet_id", "clear_tweets", "predicted_sentiment"]].head(10))