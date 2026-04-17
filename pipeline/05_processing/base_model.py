import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.database import DatabaseManager


class BaseSentimentAnalyzer(ABC):
    """Base class for sentiment analysis models.

    Subclasses declare `classificator` and implement the four abstract methods.
    The orchestration (fetch → preprocess → predict → save) lives here so every
    model gets it for free.

    Example — adding a new model:

        class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
            classificator = "NaiveBayes"

            def load_model(self): ...
            def preprocess(self, text): ...
            def predict_text(self, text, batch_size): ...
            def normalize_label(self, label): ...
    """

    model_name: str
    classificator: str

    def __init__(self) -> None:
        self.db_manager = DatabaseManager()
        self._model: Any = None

    @abstractmethod
    def load_model(self) -> Any:
        """Load and return the model or inference pipeline."""
        ...

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Apply model-specific text cleaning to a single tweet."""
        ...

    @abstractmethod
    def predict_text(self, text: str, batch_size: int) -> list[dict]:
        """Run inference on one text. Returns a list of {label, score} dicts."""
        ...

    def normalize_label(self, label: str) -> str:
        return {
            "NEGATIVE": "negativo",
            "NEUTRAL": "neutro",
            "POSITIVE": "positivo",
        }.get(label, "unknown")

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Fetch unclassified tweets, run the model, persist results, and return the rows."""
        ...

    def save(self, rows: pd.DataFrame) -> None:
        """Saves the predicted sentiment and the score into the Tweets Classification table."""
        for record in rows.itertuples():
            for info in record.predicted_sentiment:
                label = self.normalize_label(info["label"])
                score = info["score"]
                saved = self.db_manager.insert_tweets_classification(
                    tweet_id=record.id,
                    is_finance_news=1,
                    why_is_finance_news="",
                    sentiment=label,
                    why_sentiment="",
                    classificator=self.classificator,
                    score=score,
                )
                if saved:
                    print(f"Tweet ID {record.tweet_id}: {label} | {score:.4f}")
                else:
                    print(f"Tweet ID {record.tweet_id}: save failed")
