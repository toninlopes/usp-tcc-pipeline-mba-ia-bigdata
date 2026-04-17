import pandas as pd

from transformers.pipelines import pipeline
from transformers.models.bert import BertForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from .base_model import BaseSentimentAnalyzer
from shared.text_cleaner import (
    replace_urls,
    remove_emojis,
    replace_mentions,
    remove_hashtags,
    space_normalization,
    lowercase_normalization,
)


class FinBERTAnalyzer(BaseSentimentAnalyzer):
    """FinBERT-PT-BR sentiment analyzer. Extends BaseSentimentAnalyzer."""

    model_name = "lucas-leme/FinBERT-PT-BR"
    classificator = "FinBERT-PT-BR"

    def __init__(self) -> None:
        super().__init__()
        self._model = self.load_model()

    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(self.model_name)
        return pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    def preprocess(self, text: str) -> str:
        text = replace_urls(text)
        text = remove_emojis(text)
        text = replace_mentions(text)
        text = remove_hashtags(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        return text

    def predict_text(self, text: str, batch_size: int) -> list[dict]:
        return self._model(text, batch_size=batch_size)

    def run(self) -> pd.DataFrame:
        """Fetch unclassified tweets, run the model, persist results, and return the rows."""
        rows = (
            self.db_manager.query_all_tweets_with_human_but_not_finBERT_classification()
        )

        if not len(rows):
            print("No tweets found for classification.")
            return pd.DataFrame()

        self._model = self.load_model()
        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)
        rows["predicted_sentiment"] = rows["clear_tweets"].apply(
            lambda x: self.predict_text(x, batch_size=32)
        )

        return rows


if __name__ == "__main__":
    fin_bert = FinBERTAnalyzer()
    sentiments = fin_bert.run()
    sentiments.head()
