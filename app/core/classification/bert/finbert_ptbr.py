import pandas as pd
from transformers.pipelines import pipeline
from transformers.models.bert import BertForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer

from app.core.classification.bert.bert_analyzer import BertSentimentAnalyzer
from app.shared.text_cleaner import (
    space_normalization,
)


class FinBertPTBRAnalyzer(BertSentimentAnalyzer):
    """Analisador de sentimento FinBERT-PT-BR.

    Implementação concreta para o modelo lucas-leme/FinBERT-PT-BR,
    especializado no domínio financeiro em português do Brasil.

    Referência: Santos, Bianchi & Costa (2023).
    """

    model_name = "lucas-leme/FinBERT-PT-BR"
    classificator = "FinBERT-PT-BR"

    def __init__(self) -> None:
        super().__init__()
        self._model = self.load_model()

    def load_model(self):
        """Carrega o tokenizador e o modelo FinBERT-PT-BR do HuggingFace."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = BertForSequenceClassification.from_pretrained(self.model_name)
        return pipeline(task="text-classification", model=model, tokenizer=tokenizer)

    def preprocess(self, text: str) -> str:
        """Aplica etapas de limpeza adequadas ao FinBERT-PT-BR.

        Melhor configuração salva para FinBERT-PT-BR — concordância 40.5%: Normalizar espaços
        """
        text = space_normalization(text)
        return text

    def run(self) -> pd.DataFrame:
        """Busca tweets com classificação humana mas sem FinBERT, classifica e persiste.

        Returns:
            DataFrame com colunas originais do tweet mais `clear_tweets`
            e `predicted_sentiment`.
        """
        rows = self._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification()

        if rows.empty:
            print("Nenhum tweet encontrado para classificação.")
            return pd.DataFrame()

        self._model = self.load_model()
        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)
        rows["predicted_sentiment"] = rows["clear_tweets"].apply(
            lambda text: [{"label": r[0], "score": r[1]} for r in [self.predict(text)]]
        )

        return rows