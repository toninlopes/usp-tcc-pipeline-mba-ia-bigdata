from unittest.mock import MagicMock, call, patch
from collections import namedtuple
import pandas as pd
import pytest

from app.core.extraction.__main__ import run, now_utc_str


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_log_row(log_id=1, status="pending", next_token=""):
    """Cria uma linha de log simulada como namedtuple (equivalente a itertuples)."""
    LogRow = namedtuple("LogRow", ["id", "search_term", "status"])
    return LogRow(
        id=log_id,
        search_term={
            "x_user_id": "59773459",
            "from_date_time": "2025-10-01T00:00:00Z",
            "to_date_time": "2025-10-31T23:59:59Z",
        },
        status=status,
    )


def make_tweets_response(tweet_ids: list[str], next_token: str = "") -> dict:
    """Monta uma resposta simulada da API X v2."""
    return {
        "data": [
            {
                "id": tid,
                "text": f"tweet {tid}",
                "author_id": "59773459",
                "lang": "pt",
                "created_at": "2025-10-01T12:00:00Z",
                "entities": {"annotations": [], "urls": []},
                "note_tweet": {
                    "text": f"conteúdo do tweet {tid}",
                    "entities": {"hashtags": []},
                },
                "public_metrics": {
                    "like_count": 1,
                    "retweet_count": 0,
                    "reply_count": 0,
                    "quote_count": 0,
                },
                "edit_history_tweet_ids": [tid],
            }
            for tid in tweet_ids
        ],
        "meta": {
            "result_count": len(tweet_ids),
            "newest_id": tweet_ids[0] if tweet_ids else "",
            "oldest_id": tweet_ids[-1] if tweet_ids else "",
            "next_token": next_token,
        },
    }


def make_repos(pending_logs: pd.DataFrame, make_request_side_effect: list):
    """Instancia log_repo e tweet_repo já mockados."""
    log_repo = MagicMock()
    log_repo.check_connection.return_value = 0
    log_repo.query_all_pending_logs.return_value = pending_logs

    tweet_repo = MagicMock()
    tweet_repo.insert_tweet.return_value = True

    extractor = MagicMock()
    extractor.make_request.side_effect = make_request_side_effect

    return log_repo, tweet_repo, extractor


# ── now_utc_str ───────────────────────────────────────────────────────────────

class TestNowUtcStr:
    def test_returns_string(self):
        assert isinstance(now_utc_str(), str)

    def test_matches_iso8601_format(self):
        import re
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        assert re.match(pattern, now_utc_str())


# ── run — conexão ─────────────────────────────────────────────────────────────

class TestRunConnection:
    def test_exits_when_connection_fails(self):
        log_repo = MagicMock()
        log_repo.check_connection.return_value = 1

        with pytest.raises(SystemExit):
            run(log_repo=log_repo, tweet_repo=MagicMock())

    def test_returns_early_when_no_pending_logs(self):
        log_repo = MagicMock()
        log_repo.check_connection.return_value = 0
        log_repo.query_all_pending_logs.return_value = pd.DataFrame()

        tweet_repo = MagicMock()
        run(log_repo=log_repo, tweet_repo=tweet_repo)

        tweet_repo.insert_tweet.assert_not_called()


# ── run — coleta de página única ──────────────────────────────────────────────

class TestRunSinglePage:
    def test_inserts_tweets_from_single_page(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[make_tweets_response(["1", "2", "3"])],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        assert tweet_repo.insert_tweet.call_count == 3

    def test_marks_log_as_completed_on_full_success(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[make_tweets_response(["1", "2"])],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        final_update = log_repo.update_log.call_args_list[-1]
        assert final_update[0][1]["status"] == "completed"
        assert final_update[0][1]["tweets_collected"] == 2

    def test_marks_log_as_partially_completed_when_inserts_fail(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[make_tweets_response(["1", "2"])],
        )
        tweet_repo.insert_tweet.return_value = False

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        final_update = log_repo.update_log.call_args_list[-1]
        assert final_update[0][1]["status"] == "partially_completed"

    def test_records_start_time_before_collection(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[make_tweets_response(["1"])],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        first_update = log_repo.update_log.call_args_list[0]
        assert "start_time" in first_update[0][1]

    def test_skips_tweets_without_note_tweet(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        response = make_tweets_response(["1"])
        response["data"][0]["note_tweet"] = {}  # text será None

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[response],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        tweet_repo.insert_tweet.assert_not_called()


# ── run — paginação ───────────────────────────────────────────────────────────

class TestRunPagination:
    def test_follows_next_token_pagination(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        page1 = make_tweets_response(["1", "2"], next_token="token_pg2")
        page2 = make_tweets_response(["3", "4"])  # sem next_token → para

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[page1, page2],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        assert tweet_repo.insert_tweet.call_count == 4

    def test_stops_pagination_when_no_next_token(self):
        log_row = make_log_row()
        pending = pd.DataFrame([log_row._asdict()])

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[make_tweets_response(["1"])],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        assert extractor.make_request.call_count == 1


# ── run — múltiplos logs ──────────────────────────────────────────────────────

class TestRunMultipleLogs:
    def test_processes_all_pending_logs(self):
        rows = [make_log_row(log_id=1), make_log_row(log_id=2)]
        pending = pd.DataFrame([r._asdict() for r in rows])

        log_repo, tweet_repo, extractor = make_repos(
            pending_logs=pending,
            make_request_side_effect=[
                make_tweets_response(["1"]),
                make_tweets_response(["2"]),
            ],
        )

        with patch("app.core.extraction.__main__.TwitterAPIExtractor", return_value=extractor):
            run(log_repo=log_repo, tweet_repo=tweet_repo)

        assert tweet_repo.insert_tweet.call_count == 2
        assert log_repo.update_log.call_count == 4  # start + end para cada log