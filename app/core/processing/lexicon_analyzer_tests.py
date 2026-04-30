from typing import Dict, Tuple
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from app.core.processing.lexicon_analyzer import LexiconSentimentAnalyzer


# ── Implementação mínima para testar a classe abstrata ────────────────────────

class ConcreteLexiconAnalyzer(LexiconSentimentAnalyzer):
    classificator = "TestLexicon"
    model_name = "test"

    def load_model(self) -> Dict[str, int]:
        return {"bom": 1, "ótimo": 1, "ruim": -1, "péssimo": -1}

    def preprocess(self, text: str) -> str:
        return text.lower()

    def predict(self, text: str) -> Tuple[str, float]:
        tokens = text.split()
        lexicon = self._model
        score = sum(lexicon.get(t, 0) for t in tokens)
        if score > 0:
            return "positivo", 0.8
        elif score < 0:
            return "negativo", 0.8
        return "neutro", 0.0


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    with patch("app.shared.database.load_dotenv"):
        instance = ConcreteLexiconAnalyzer.__new__(ConcreteLexiconAnalyzer)
        instance._tweet_repo = MagicMock()
        instance._classification_repo = MagicMock()
        instance._model = {"bom": 1, "ótimo": 1, "ruim": -1, "péssimo": -1}
        return instance


@pytest.fixture
def tweets_df():
    return pd.DataFrame([
        {"id": 1, "note_tweet": "resultado bom", "sentiment": "positivo",
         "is_finance_tweet": 1, "has_human_classification": True},
        {"id": 2, "note_tweet": "resultado ruim", "sentiment": "negativo",
         "is_finance_tweet": 1, "has_human_classification": True},
        {"id": 3, "note_tweet": "resultado neutro", "sentiment": "neutro",
         "is_finance_tweet": 1, "has_human_classification": False},
    ])


# ── run — sem tweets ──────────────────────────────────────────────────────────

class TestRunEmpty:
    def test_returns_empty_dataframe_when_repo_empty(self, analyzer):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = pd.DataFrame()
        result = analyzer.run()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_dataframe_when_no_human_classification(self, analyzer):
        df = pd.DataFrame([{
            "id": 1, "note_tweet": "texto",
            "is_finance_tweet": 1, "has_human_classification": False,
        }])
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = df
        result = analyzer.run()
        assert result.empty


# ── run — filtragem ───────────────────────────────────────────────────────────

class TestRunFiltering:
    def test_filters_tweets_without_human_classification(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        result = analyzer.run()
        assert len(result) == 2
        assert all(result["has_human_classification"] == True)

    def test_does_not_modify_original_dataframe(self, analyzer, tweets_df):
        original_len = len(tweets_df)
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        analyzer.run()
        assert len(tweets_df) == original_len


# ── run — colunas ─────────────────────────────────────────────────────────────

class TestRunColumns:
    def test_adds_clear_tweets_column(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        result = analyzer.run()
        assert "clear_tweets" in result.columns

    def test_adds_predicted_sentiment_column(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        result = analyzer.run()
        assert "predicted_sentiment" in result.columns

    def test_predicted_sentiment_contains_label_and_score(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        result = analyzer.run()
        first = result.iloc[0]["predicted_sentiment"][0]
        assert "label" in first
        assert "score" in first


# ── run — classificação ───────────────────────────────────────────────────────

class TestRunClassification:
    def test_calls_preprocess_for_each_tweet(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        call_count = [0]
        original = analyzer.preprocess

        def counting_preprocess(text):
            call_count[0] += 1
            return original(text)

        analyzer.preprocess = counting_preprocess
        analyzer.run()
        assert call_count[0] == 2  # apenas tweets com classificação humana

    def test_calls_predict_for_each_tweet(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_classification.return_value = tweets_df
        call_count = [0]
        original = analyzer.predict

        def counting_predict(text):
            call_count[0] += 1
            return original(text)

        analyzer.predict = counting_predict
        analyzer.run()
        assert call_count[0] == 2