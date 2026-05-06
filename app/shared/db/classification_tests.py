from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
import pytest

from app.shared.db.classification import ClassificationRepository
from app.shared.test_helpers import make_repo, mock_get_connection


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def repo():
    return make_repo(ClassificationRepository)


@pytest.fixture
def cursor():
    return MagicMock()


@pytest.fixture
def classification_row():
    """Linha simulada retornada pelo banco para tweets_classification."""
    return (1, 42, "positivo", "Tom otimista", 1, "Menciona resultado financeiro", "Humano", None)


# ── insert_tweets_classification ──────────────────────────────────────────────

class TestInsertTweetsClassification:
    def test_returns_true_on_success(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        result = repo.insert_tweets_classification(
            tweet_id=42,
            is_finance_news=1,
            why_is_finance_news="Menciona resultado financeiro",
            sentiment="positivo",
            why_sentiment="Tom otimista",
            classificator="Humano",
        )
        assert result is True

    def test_commits_on_success(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.insert_tweets_classification(
            tweet_id=42,
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="neutro",
            why_sentiment="",
            classificator="Humano",
        )
        conn.commit.assert_called_once()

    def test_passes_score_to_query(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.insert_tweets_classification(
            tweet_id=42,
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="FinBERT-PT-BR",
            score=0.97,
        )
        args = cursor.execute.call_args[0][1]
        assert args[-1] == 0.97

    def test_score_defaults_to_none(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.insert_tweets_classification(
            tweet_id=42,
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="Humano",
        )
        args = cursor.execute.call_args[0][1]
        assert args[-1] is None

    def test_returns_false_and_rollbacks_on_exception(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        cursor.execute.side_effect = Exception("db error")
        result = repo.insert_tweets_classification(
            tweet_id=42,
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="Humano",
        )
        assert result is False
        conn.rollback.assert_called_once()

    def test_normalizes_numpy_tweet_id(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        numpy_id = np.int64(42)
        repo.insert_tweets_classification(
            tweet_id=numpy_id,
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="Humano",
        )
        args = cursor.execute.call_args[0][1]
        assert args[0] == 42
        assert isinstance(args[0], int)

    def test_normalizes_list_tweet_id(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.insert_tweets_classification(
            tweet_id=[42],
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="Humano",
        )
        args = cursor.execute.call_args[0][1]
        assert args[0] == 42
        assert isinstance(args[0], int)

    def test_casts_is_finance_news_to_int(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.insert_tweets_classification(
            tweet_id=42,
            is_finance_news=True,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="Humano",
        )
        args = cursor.execute.call_args[0][1]
        assert args[1] == 1
        assert isinstance(args[1], int)


# ── query_tweets_classification_by_id ────────────────────────────────────────

class TestQueryTweetsClassificationById:
    def test_returns_dataframe(self, repo, cursor, classification_row):
        cursor.fetchall.return_value = [classification_row]
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(42)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, classification_row):
        cursor.fetchall.return_value = [classification_row]
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(42)
        expected_columns = [
            "id", "tweet_id", "sentiment", "why_sentiment",
            "is_finance_news", "why_is_finance_news", "classificator", "score",
        ]
        assert list(result.columns) == expected_columns

    def test_returns_correct_values(self, repo, cursor, classification_row):
        cursor.fetchall.return_value = [classification_row]
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(42)
        assert result.iloc[0]["sentiment"] == "positivo"
        assert result.iloc[0]["classificator"] == "Humano"
        assert result.iloc[0]["tweet_id"] == 42

    def test_returns_empty_dataframe_on_no_results(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(42)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(42)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_normalizes_numpy_tweet_id(self, repo, cursor, classification_row):
        cursor.fetchall.return_value = [classification_row]
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(np.int64(42))
        args = cursor.execute.call_args[0][1]
        assert args == (42,)
        assert isinstance(args[0], int)

    def test_normalizes_list_tweet_id(self, repo, cursor, classification_row):
        cursor.fetchall.return_value = [classification_row]
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id([42])
        args = cursor.execute.call_args[0][1]
        assert args == (42,)
        assert isinstance(args[0], int)

    def test_multiple_rows_returned(self, repo, cursor, classification_row):
        cursor.fetchall.return_value = [classification_row, classification_row]
        mock_get_connection(repo, cursor)
        result = repo.query_tweets_classification_by_id(42)
        assert len(result) == 2

# ── query_classification_pairs ────────────────────────────────────────────────

class TestQueryClassificationPairs:
    @pytest.fixture
    def pair_row(self):
        return (1, "positivo", "positivo")

    def test_returns_dataframe(self, repo, cursor, pair_row):
        cursor.fetchall.return_value = [pair_row]
        mock_get_connection(repo, cursor)
        result = repo.query_classification_pairs("FinBERT-PT-BR")
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, pair_row):
        cursor.fetchall.return_value = [pair_row]
        mock_get_connection(repo, cursor)
        result = repo.query_classification_pairs("FinBERT-PT-BR")
        assert list(result.columns) == ["tweet_id", "human_label", "model_label"]

    def test_returns_correct_values(self, repo, cursor, pair_row):
        cursor.fetchall.return_value = [pair_row]
        mock_get_connection(repo, cursor)
        result = repo.query_classification_pairs("FinBERT-PT-BR")
        assert result.iloc[0]["tweet_id"] == 1
        assert result.iloc[0]["human_label"] == "positivo"
        assert result.iloc[0]["model_label"] == "positivo"

    def test_default_split_is_teste(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_classification_pairs("FinBERT-PT-BR")
        params = cursor.execute.call_args[0][1]
        assert params == ("FinBERT-PT-BR", "teste")

    def test_passes_custom_split(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_classification_pairs("SentiLex-PT", split="treino")
        params = cursor.execute.call_args[0][1]
        assert params == ("SentiLex-PT", "treino")

    def test_no_split_filter_when_none(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_classification_pairs("FinBERT-PT-BR", split=None)
        params = cursor.execute.call_args[0][1]
        assert params == ("FinBERT-PT-BR",)

    def test_split_clause_absent_when_none(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_classification_pairs("FinBERT-PT-BR", split=None)
        query = cursor.execute.call_args[0][0]
        assert "h.split" not in query

    def test_split_clause_present_when_provided(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_classification_pairs("FinBERT-PT-BR", split="teste")
        query = cursor.execute.call_args[0][0]
        assert "h.split" in query

    def test_returns_empty_dataframe_on_no_results(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        result = repo.query_classification_pairs("FinBERT-PT-BR")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_classification_pairs("FinBERT-PT-BR")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_multiple_pairs_returned(self, repo, cursor):
        cursor.fetchall.return_value = [
            (1, "positivo", "positivo"),
            (2, "negativo", "neutro"),
            (3, "neutro", "neutro"),
        ]
        mock_get_connection(repo, cursor)
        result = repo.query_classification_pairs("FinBERT-PT-BR")
        assert len(result) == 3