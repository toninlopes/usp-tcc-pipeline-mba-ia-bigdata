from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.database_tweets import DatabaseTweetsQuery

from base_model import BaseSentimentAnalyzer
from shared.text_cleaner import (
    replace_emojis_with_codes,
    remove_hashtags,
    replace_mentions,
    replace_urls,
    space_normalization,
)


_FINE_TUNED_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "bert-timbau-sentiment"
)


class BERTimbauAnalyzer(BaseSentimentAnalyzer):
    """BERTimbau fine-tuned sentiment analyzer.

    Requer um modelo treinado em models/bert-timbau-sentiment/ produzido por
    bert_timbau_fine_tuner. O processing_dashboard detecta a ausência do modelo
    antes de instanciar esta classe, mas mantemos a verificação como defesa.
    """

    model_name = str(_FINE_TUNED_PATH)
    classificator = "BERTimbau"

    def __init__(self) -> None:
        super().__init__()
        if not _FINE_TUNED_PATH.exists():
            raise RuntimeError(
                f"Modelo fine-tuned não encontrado em {_FINE_TUNED_PATH}. "
                f"Execute: python -m pipeline.05_processing.bert_timbau_fine_tuner"
            )
        self._model = self.load_model()
        self._db_tweets = DatabaseTweetsQuery(self.db_manager)

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        return pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    def preprocess(self, text: str) -> str:
        """Mesmo preprocessing aplicado no fine-tuning — sem lowercase_normalization."""
        text = replace_urls(text)
        text = replace_emojis_with_codes(text)
        text = replace_mentions(text)
        text = remove_hashtags(text)
        text = space_normalization(text)
        return text

    def normalize_label(self, label: str) -> str:
        """O modelo fine-tuned emite labels em português via id2label. Preserva
        esses valores e mantém fallback para labels em inglês por compatibilidade."""
        if label in {"positivo", "negativo", "neutro"}:
            return label
        return super().normalize_label(label)

    def predict_text(self, text: str, batch_size: int) -> list[dict]:
        return self._model(text, batch_size=batch_size)

    def run(self) -> pd.DataFrame:
        """Fetch, preprocess, predict. Save é chamado pelo dashboard."""
        rows = self._db_tweets.fetch_tweets_with_human_classification()
        if not len(rows):
            print("No tweets found for classification.")
            return pd.DataFrame()

        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)

        new_rows = rows.head(
            1000
        )  # Limite para teste rápido; remova para processar tudo

        new_rows["predicted_sentiment"] = new_rows["clear_tweets"].apply(
            lambda x: self.predict_text(x, batch_size=32)
        )
        return new_rows


if __name__ == "__main__":
    analyzer = BERTimbauAnalyzer()
    sentiments = analyzer.run()
    print(sentiments.head())
