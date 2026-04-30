from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from app.shared.db_tweets import TweetsRepository
from app.shared.db_classification import ClassificationRepository


class BaseSentimentAnalyzer(ABC):
    """Contrato público para analisadores de sentimento.

    Define o pipeline fetch → preprocess → predict → save, sem assumir
    a tecnologia subjacente (BERT, léxico, etc.).

    Para adicionar um novo modelo:

        class MyAnalyzer(BaseSentimentAnalyzer):
            classificator = "MyModel"

            def preprocess(self, text: str) -> str: ...
            def predict(self, text: str) -> Tuple[str, float]: ...
            def run(self) -> pd.DataFrame: ...
    """

    classificator: str

    def __init__(self) -> None:
        self._tweet_repo = TweetsRepository()
        self._classification_repo = ClassificationRepository()

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Aplica limpeza textual específica do modelo a um único tweet."""
        ...

    @abstractmethod
    def predict(self, text: str) -> Tuple[str, float]:
        """Executa inferência em um texto.

        Returns:
            Tupla (label_pt, score) onde label_pt é 'positivo', 'negativo' ou 'neutro'.
        """
        ...

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Busca tweets não classificados, executa o modelo e persiste os resultados."""
        ...

    def normalize_label(self, label: str) -> str:
        """Converte rótulos em inglês do HuggingFace para português."""
        return {
            "NEGATIVE": "negativo",
            "NEUTRAL": "neutro",
            "POSITIVE": "positivo",
        }.get(label, "unknown")

    def save(self, rows: pd.DataFrame) -> None:
        """Persiste as predições em `tweets_classification`."""
        for record in rows.itertuples():
            for prediction in record.predicted_sentiment:
                label = prediction["label"]  # predict() já retorna em português
                score = prediction["score"]
                saved = self._classification_repo.insert_tweets_classification(
                    tweet_id=record.id,
                    is_finance_news=1,
                    why_is_finance_news="",
                    sentiment=label,
                    why_sentiment="",
                    classificator=self.classificator,
                    score=score,
                )
                if saved:
                    print(f"Tweet ID {record.id}: {label} | {score:.4f}")
                else:
                    print(f"Tweet ID {record.id}: falha ao salvar")