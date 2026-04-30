from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# Bloqueia HuggingFace antes de qualquer import do módulo
import sys
from unittest.mock import MagicMock

for _mod in [
    "transformers",
    "transformers.pipelines",
    "transformers.models",
    "transformers.models.bert",
    "transformers.models.auto",
    "transformers.models.auto.tokenization_auto",
]:
    sys.modules.setdefault(_mod, MagicMock())

from app.core.processing.finbert_ptbr import FinBertPTBRAnalyzer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.return_value = [{"label": "POSITIVE", "score": 0.95}]
    return model


@pytest.fixture
def analyzer(mock_model):
    """FinBertPTBRAnalyzer com DB e modelo mockados — ignora __init__."""
    instance = FinBertPTBRAnalyzer.__new__(FinBertPTBRAnalyzer)
    instance._tweet_repo = MagicMock()
    instance._classification_repo = MagicMock()
    instance._model = mock_model
    instance.load_model = MagicMock(return_value=mock_model)
    return instance


@pytest.fixture
def tweets_df():
    return pd.DataFrame([
        {"id": 1, "tweet_id": "abc", "note_tweet": "Alta do IBOV hoje https://t.co/x @Info #IBOV 🚨", "sentiment": "positivo", "is_finance_tweet": 1, "has_human_classification": True, "has_finbert_classification": False},
        {"id": 2, "tweet_id": "def", "note_tweet": "Queda acentuada do mercado", "sentiment": "negativo", "is_finance_tweet": 1, "has_human_classification": True, "has_finbert_classification": False},
    ])


# ── Atributos de classe ───────────────────────────────────────────────────────

class TestClassAttributes:
    def test_model_name(self):
        assert FinBertPTBRAnalyzer.model_name == "lucas-leme/FinBERT-PT-BR"

    def test_classificator(self):
        assert FinBertPTBRAnalyzer.classificator == "FinBERT-PT-BR"


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_removes_url(self, analyzer):
        result = analyzer.preprocess("Veja https://t.co/abc")
        assert "https://" not in result
        assert "[URL]" in result

    def test_removes_emoji(self, analyzer):
        result = analyzer.preprocess("🚨 Alerta!")
        assert "🚨" not in result

    def test_replaces_mention(self, analyzer):
        result = analyzer.preprocess("Segundo @InfoMoney")
        assert "@InfoMoney" not in result
        assert "[MENTION]" in result

    def test_removes_hashtag_symbol(self, analyzer):
        result = analyzer.preprocess("Alta do #IBOV")
        assert "#" not in result
        assert "IBOV" in result

    def test_collapses_spaces(self, analyzer):
        result = analyzer.preprocess("texto   com   espaços")
        assert "  " not in result

    def test_lowercases_regular_words(self, analyzer):
        result = analyzer.preprocess("Alta do Mercado")
        assert "mercado" in result

    def test_preserves_ticker(self, analyzer):
        result = analyzer.preprocess("Comprei PETR4")
        assert "PETR4" in result

    def test_preserves_financial_entity(self, analyzer):
        result = analyzer.preprocess("Decisão do COPOM")
        assert "COPOM" in result


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def test_returns_empty_dataframe_when_no_tweets(self, analyzer):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = pd.DataFrame()
        result = analyzer.run()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe_with_clear_tweets_column(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df
        result = analyzer.run()
        assert "clear_tweets" in result.columns

    def test_returns_dataframe_with_predicted_sentiment_column(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df
        result = analyzer.run()
        assert "predicted_sentiment" in result.columns

    def test_predicted_sentiment_contains_label_and_score(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df
        result = analyzer.run()
        first = result.iloc[0]["predicted_sentiment"][0]
        assert "label" in first
        assert "score" in first

    def test_calls_preprocess_for_each_tweet(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df

        call_count = [0]
        original = analyzer.preprocess

        def counting_preprocess(text):
            call_count[0] += 1
            return original(text)

        analyzer.preprocess = counting_preprocess
        analyzer.run()

        assert call_count[0] == len(tweets_df)

    def test_reloads_model_on_run(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df
        analyzer.run()
        analyzer.load_model.assert_called_once()