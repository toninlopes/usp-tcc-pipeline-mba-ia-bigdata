from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
import pytest

from app.shared.db.tweets import TweetsRepository
from app.shared.test_helpers import make_repo, mock_get_connection


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def repo():
    return make_repo(TweetsRepository)


@pytest.fixture
def tweet_row():
    return (
        1, "1234567890", "59773459",
        "Mercado em alta nesta semana.",
        "2025-11-01T12:00:00Z", 42, None, None,
        "positivo", 1,
    )


@pytest.fixture
def tweet_row_with_flags(tweet_row):
    return tweet_row + (True,)


@pytest.fixture
def tweet_row_with_all_flags(tweet_row):
    return tweet_row + (True, False)


@pytest.fixture
def tweet_data():
    return {
        "tweet_id": "1234567890",
        "username": "59773459",
        "note_tweet": "Mercado em alta nesta semana.",
        "created_at": "2025-11-01T12:00:00Z",
        "likes": 42,
        "hashtags": None,
        "tweet": {"id": "1234567890"},
    }


EXPECTED_BASE_COLUMNS = [
    "id", "tweet_id", "username", "note_tweet",
    "created_at", "likes", "hashtags", "tweet",
    "sentiment", "is_finance_tweet",
]


# ── insert_tweet ──────────────────────────────────────────────────────────────

class TestInsertTweet:
    def test_returns_true_on_success(self, repo, cursor, tweet_data):
        conn = mock_get_connection(repo, cursor)
        assert repo.insert_tweet(tweet_data) is True

    def test_commits_on_success(self, repo, cursor, tweet_data):
        conn = mock_get_connection(repo, cursor)
        repo.insert_tweet(tweet_data)
        conn.commit.assert_called_once()

    def test_returns_false_and_rollbacks_on_exception(self, repo, cursor, tweet_data):
        conn = mock_get_connection(repo, cursor)
        cursor.execute.side_effect = Exception("db error")
        assert repo.insert_tweet(tweet_data) is False
        conn.rollback.assert_called_once()

    def test_passes_none_for_missing_hashtags(self, repo, cursor, tweet_data):
        conn = mock_get_connection(repo, cursor)
        tweet_data["hashtags"] = None
        repo.insert_tweet(tweet_data)
        args = cursor.execute.call_args[0][1]
        assert args[5] is None

    def test_passes_none_for_missing_tweet_json(self, repo, cursor, tweet_data):
        conn = mock_get_connection(repo, cursor)
        tweet_data["tweet"] = None
        repo.insert_tweet(tweet_data)
        args = cursor.execute.call_args[0][1]
        assert args[6] is None


# ── query_all_tweets ──────────────────────────────────────────────────────────

class TestQueryAllTweets:
    def test_returns_dataframe(self, repo, cursor, tweet_row):
        cursor.fetchall.return_value = [tweet_row]
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets()
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, tweet_row):
        cursor.fetchall.return_value = [tweet_row]
        mock_get_connection(repo, cursor)
        assert list(repo.query_all_tweets().columns) == EXPECTED_BASE_COLUMNS

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── query_all_tweets_with_human_classification ────────────────────────────────

class TestQueryAllTweetsWithHumanClassification:
    def test_returns_dataframe(self, repo, cursor, tweet_row_with_flags):
        cursor.fetchall.return_value = [tweet_row_with_flags]
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets_with_human_classification()
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, tweet_row_with_flags):
        cursor.fetchall.return_value = [tweet_row_with_flags]
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets_with_human_classification()
        assert list(result.columns) == EXPECTED_BASE_COLUMNS + ["has_human_classification"]

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets_with_human_classification()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── query_all_tweets_with_human_but_not_finbert_classification ────────────────

class TestQueryAllTweetsWithHumanButNotFinbert:
    def test_returns_dataframe(self, repo, cursor, tweet_row_with_all_flags):
        cursor.fetchall.return_value = [tweet_row_with_all_flags]
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets_with_human_but_not_finbert_classification()
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, tweet_row_with_all_flags):
        cursor.fetchall.return_value = [tweet_row_with_all_flags]
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets_with_human_but_not_finbert_classification()
        assert list(result.columns) == EXPECTED_BASE_COLUMNS + [
            "has_human_classification", "has_finbert_classification"
        ]

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_all_tweets_with_human_but_not_finbert_classification()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── update_sentiment ──────────────────────────────────────────────────────────

class TestUpdateSentiment:
    def test_returns_true_on_success(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        assert repo.update_sentiment(1, "positivo") is True

    def test_commits_on_success(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.update_sentiment(1, "positivo")
        conn.commit.assert_called_once()

    def test_passes_correct_args(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.update_sentiment(1, "negativo")
        args = cursor.execute.call_args[0][1]
        assert args == ("negativo", 1)

    def test_returns_false_and_rollbacks_on_exception(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        cursor.execute.side_effect = Exception("db error")
        assert repo.update_sentiment(1, "positivo") is False
        conn.rollback.assert_called_once()


# ── update_is_finance_news ────────────────────────────────────────────────────

class TestUpdateIsFinanceNews:
    def test_returns_true_on_success(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        assert repo.update_is_finance_news(1, 1) is True

    def test_commits_on_success(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.update_is_finance_news(1, 1)
        conn.commit.assert_called_once()

    def test_nulls_sentiment_when_not_finance(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.update_is_finance_news(1, 0)
        query = cursor.execute.call_args[0][0]
        assert "sentiment = NULL" in query

    def test_does_not_null_sentiment_when_finance(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        repo.update_is_finance_news(1, 1)
        query = cursor.execute.call_args[0][0]
        assert "sentiment = NULL" not in query

    def test_returns_false_and_rollbacks_on_exception(self, repo, cursor):
        conn = mock_get_connection(repo, cursor)
        cursor.execute.side_effect = Exception("db error")
        assert repo.update_is_finance_news(1, 1) is False
        conn.rollback.assert_called_once()