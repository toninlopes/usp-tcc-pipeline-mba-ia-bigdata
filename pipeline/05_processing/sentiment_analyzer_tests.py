import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── stub heavy dependencies before the module executes ───────────────────────
_mock_db_instance = MagicMock()
sys.modules.setdefault("shared", MagicMock())
sys.modules["shared.database"] = MagicMock(
    DatabaseManager=MagicMock(return_value=_mock_db_instance)
)
sys.modules.setdefault("transformers", MagicMock())
# ─────────────────────────────────────────────────────────────────────────────

_path = Path(__file__).parent / "sentiment_analyzer.py"
_spec = importlib.util.spec_from_file_location("sentiment_analyzer", _path)
assert _spec is not None and _spec.loader is not None
_sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sa)  # type: ignore[union-attr]


def _make_df(rows: list[dict]) -> pd.DataFrame:
    cols = [
        "id", "tweet_id", "username", "note_tweet", "created_at",
        "likes", "hashtags", "tweet", "sentiment", "is_finance_tweet",
        "has_human_classification", "has_finbert_classification",
    ]
    return pd.DataFrame({c: [r.get(c) for r in rows] for c in cols})


_SAMPLE_ROW = {
    "id": 7,
    "tweet_id": "2021735781283356906",
    "note_tweet": "Alta do #IBOV https://t.co/abc",
    "username": "InfoMoney",
    "created_at": None, "likes": 0, "hashtags": None,
    "tweet": None, "sentiment": None, "is_finance_tweet": 1,
    "has_human_classification": True, "has_finbert_classification": False,
}


@pytest.fixture(autouse=True)
def reset_db_mock():
    _mock_db_instance.reset_mock()
    _sa.db_manager = _mock_db_instance
    yield


class TestRunEarlyExit:
    def test_no_tweets_skips_insert(self):
        _sa.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = pd.DataFrame()
        _sa.run()
        _sa.db_manager.insert_tweets_classification.assert_not_called()

    def test_no_tweets_skips_model_loading(self):
        _sa.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = pd.DataFrame()
        with patch.object(_sa, "load_model_as_pipeline") as mock_load:
            _sa.run()
        mock_load.assert_not_called()


class TestLabelConversion:
    @pytest.mark.parametrize("label,expected", [
        ("NEGATIVE", "negativo"),
        ("NEUTRAL", "neutro"),
        ("POSITIVE", "positivo"),
        ("UNKNOWN_LABEL", "unknown"),
    ])
    def test_finbert_label_mapped_to_portuguese(self, label, expected):
        mapping = {"NEGATIVE": "negativo", "NEUTRAL": "neutro", "POSITIVE": "positivo"}
        assert mapping.get(label, "unknown") == expected


class TestRun:
    def _setup(self, label="POSITIVE", score=0.99):
        df = _make_df([_SAMPLE_ROW])
        _sa.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = df
        _sa.db_manager.insert_tweets_classification.return_value = True
        mock_pipeline = MagicMock(return_value=[{"label": label, "score": score}])
        return mock_pipeline

    def test_uses_db_primary_key_not_twitter_snowflake(self):
        mock_pipeline = self._setup(label="POSITIVE", score=0.99)
        with patch.object(_sa, "load_model_as_pipeline", return_value=mock_pipeline):
            _sa.run()

        _sa.db_manager.insert_tweets_classification.assert_called_once_with(
            tweet_id=7,
            is_finance_news=1,
            why_is_finance_news="",
            sentiment="positivo",
            why_sentiment="",
            classificator="FinBERT-PT-BR",
            score=0.99,
        )

    def test_clean_applied_to_note_tweet_before_prediction(self):
        mock_pipeline = self._setup()
        mock_clean = MagicMock(return_value="cleaned text")
        with patch.object(_sa, "load_model_as_pipeline", return_value=mock_pipeline), \
             patch.object(_sa, "clean", mock_clean):
            _sa.run()

        mock_clean.assert_called_once_with("Alta do #IBOV https://t.co/abc")
        mock_pipeline.assert_called_once_with("cleaned text", batch_size=32)

    def test_custom_batch_size_forwarded_to_pipeline(self):
        mock_pipeline = self._setup()
        mock_clean = MagicMock(return_value="cleaned")
        with patch.object(_sa, "load_model_as_pipeline", return_value=mock_pipeline), \
             patch.object(_sa, "clean", mock_clean):
            _sa.run(batch_size=8)

        mock_pipeline.assert_called_once_with("cleaned", batch_size=8)

    def test_classificator_constant_written_to_db(self):
        mock_pipeline = self._setup()
        with patch.object(_sa, "load_model_as_pipeline", return_value=mock_pipeline):
            _sa.run()

        _, kwargs = _sa.db_manager.insert_tweets_classification.call_args
        assert kwargs["classificator"] == "FinBERT-PT-BR"
