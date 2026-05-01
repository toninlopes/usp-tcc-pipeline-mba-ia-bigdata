import sys
from abc import abstractmethod
from typing import Dict, Tuple

import pandas as pd

from app.core.processing.base_analyzer import BaseSentimentAnalyzer

BAR_WIDTH = 40


class LexiconSentimentAnalyzer(BaseSentimentAnalyzer):
    """Base para analisadores de sentimento baseados em léxicos.

    Extrai o pipeline comum de run() — busca, pré-processamento,
    classificação com barra de progresso — deixando para as subclasses
    apenas a lógica de carregamento e pontuação do léxico específico.

    Para adicionar um novo léxico:

        class MyLexiconAnalyzer(LexiconSentimentAnalyzer):
            classificator = "MyLexicon"

            def load_model(self) -> Dict[str, int]: ...
            def preprocess(self, text: str) -> str: ...
            def predict(self, text: str) -> Tuple[str, float]: ...
    """

    @abstractmethod
    def load_model(self) -> Dict[str, int]:
        """Carrega o léxico e retorna dict {palavra: polaridade}."""
        ...

    def run(self) -> pd.DataFrame:
        """Pipeline comum: busca tweets, classifica e retorna resultados.

        Seleciona tweets financeiros com anotação humana para permitir
        comparação direta com o gold standard na avaliação.
        """
        rows = self._tweet_repo.query_all_tweets_with_human_classification()

        if rows.empty:
            print(f"[{self.classificator}] Nenhum tweet encontrado.")
            return pd.DataFrame()

        rows = rows[rows["has_human_classification"] == True].reset_index(drop=True)

        if rows.empty:
            print(f"[{self.classificator}] Nenhum tweet com anotação humana.")
            return pd.DataFrame()

        print(f"[{self.classificator}] Pré-processando {len(rows)} tweets...")
        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)

        print(f"[{self.classificator}] Classificando {len(rows)} tweets...")
        total = len(rows)
        predictions = []

        for i, text in enumerate(rows["clear_tweets"], start=1):
            label, score = self.predict(text)
            predictions.append([{"label": label, "score": score}])
            pct = i / total
            filled = int(BAR_WIDTH * pct)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            sys.stdout.write(f"\r  [{bar}] {pct:5.1%}  {i}/{total}")
            sys.stdout.flush()

        sys.stdout.write("\n")
        rows["predicted_sentiment"] = predictions
        return rows