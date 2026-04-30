from unittest.mock import MagicMock, patch
from typing import Tuple
import pandas as pd
import pytest

from app.core.processing.base_analyzer import BaseSentimentAnalyzer


# ── Implementação mínima para testar a classe abstrata ────────────────────────

class ConcreteAnalyzer(BaseSentimentAnalyzer):
    classificator = "TestModel"

    def preprocess(self, text: str) -> str:
        return text.lower()

    def predict(self, text: str) -> Tuple[str, float]:
        return "positivo", 0.9

    def run(self) -> pd.DataFrame:
        return pd.DataFrame()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    with patch("app.shared.database.load_dotenv"):
        return ConcreteAnalyzer()


@pytest.fixture
def mock_classification_repo(analyzer):
    analyzer._classification_repo = MagicMock()
    analyzer._classification_repo.insert_tweets_classification.return_value = True
    return analyzer._classification_repo


@pytest.fixture
def rows_df():
    return pd.DataFrame([
        {
            "id": 1,
            "note_tweet": "Mercado em alta",
            "predicted_sentiment": [{"label": "positivo", "score": 0.95}],
        },
        {
            "id": 2,
            "note_tweet": "Queda brusca",
            "predicted_sentiment": [{"label": "negativo", "score": 0.88}],
        },
    ])


# ── normalize_label ───────────────────────────────────────────────────────────

class TestNormalizeLabel:
    def test_positive_maps_to_positivo(self, analyzer):
        assert analyzer.normalize_label("POSITIVE") == "positivo"

    def test_negative_maps_to_negativo(self, analyzer):
        assert analyzer.normalize_label("NEGATIVE") == "negativo"

    def test_neutral_maps_to_neutro(self, analyzer):
        assert analyzer.normalize_label("NEUTRAL") == "neutro"

    def test_unknown_label_returns_unknown(self, analyzer):
        assert analyzer.normalize_label("OUTROS") == "unknown"

    def test_empty_string_returns_unknown(self, analyzer):
        assert analyzer.normalize_label("") == "unknown"

    def test_lowercase_label_returns_unknown(self, analyzer):
        # Rótulos do HuggingFace são sempre uppercase
        assert analyzer.normalize_label("positive") == "unknown"


# ── save ──────────────────────────────────────────────────────────────────────

class TestSave:
    def test_calls_insert_for_each_row(self, analyzer, mock_classification_repo, rows_df):
        analyzer.save(rows_df)
        assert mock_classification_repo.insert_tweets_classification.call_count == 2

    def test_passes_correct_tweet_id(self, analyzer, mock_classification_repo, rows_df):
        analyzer.save(rows_df)
        calls = mock_classification_repo.insert_tweets_classification.call_args_list
        assert calls[0][1]["tweet_id"] == 1
        assert calls[1][1]["tweet_id"] == 2

    def test_passes_label_directly(self, analyzer, mock_classification_repo, rows_df):
        analyzer.save(rows_df)
        calls = mock_classification_repo.insert_tweets_classification.call_args_list
        assert calls[0][1]["sentiment"] == "positivo"
        assert calls[1][1]["sentiment"] == "negativo"

    def test_passes_correct_score(self, analyzer, mock_classification_repo, rows_df):
        analyzer.save(rows_df)
        calls = mock_classification_repo.insert_tweets_classification.call_args_list
        assert calls[0][1]["score"] == pytest.approx(0.95)
        assert calls[1][1]["score"] == pytest.approx(0.88)

    def test_passes_classificator(self, analyzer, mock_classification_repo, rows_df):
        analyzer.save(rows_df)
        calls = mock_classification_repo.insert_tweets_classification.call_args_list
        for call in calls:
            assert call[1]["classificator"] == "TestModel"

    def test_handles_multiple_predictions_per_row(self, analyzer, mock_classification_repo):
        rows = pd.DataFrame([{
            "id": 1,
            "note_tweet": "tweet",
            "predicted_sentiment": [
                {"label": "POSITIVE", "score": 0.7},
                {"label": "NEUTRAL", "score": 0.3},
            ],
        }])
        analyzer.save(rows)
        assert mock_classification_repo.insert_tweets_classification.call_count == 2