import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Add project root so shared.* is importable
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

# Block HuggingFace / transformers (heavy downloads) before any module is loaded
for _m in [
    "transformers",
    "transformers.pipelines",
    "transformers.models",
    "transformers.models.bert",
    "transformers.models.auto",
    "transformers.models.auto.tokenization_auto",
]:
    sys.modules.setdefault(_m, MagicMock())

# Block DB connection; DatabaseManager becomes a plain MagicMock class
sys.modules.setdefault("shared.database", MagicMock())

# Register a synthetic parent package so `.base_model` relative imports resolve
_PKG = "_pipeline_05_processing"
_pkg = types.ModuleType(_PKG)
_pkg.__path__ = [str(_HERE)]
_pkg.__package__ = _PKG
sys.modules.setdefault(_PKG, _pkg)


def _load(filename: str) -> types.ModuleType:
    name = f"{_PKG}.{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("base_model.py")
FinBERTAnalyzer = _load("fin_bert_timbau.py").FinBERTAnalyzer


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model():
    """Fake HuggingFace pipeline that returns a predictable POSITIVE result."""
    model = MagicMock()
    model.return_value = [{"label": "POSITIVE", "score": 0.95}]
    return model


@pytest.fixture
def analyzer(mock_model):
    """FinBERTAnalyzer with DB and model mocked — bypasses __init__."""
    instance = FinBERTAnalyzer.__new__(FinBERTAnalyzer)
    instance.db_manager = MagicMock()
    instance._model = mock_model
    instance.load_model = MagicMock(return_value=mock_model)
    return instance


# ── Class attributes ──────────────────────────────────────────────────────────


class TestClassAttributes:
    def test_classificator(self):
        assert FinBERTAnalyzer.classificator == "FinBERT-PT-BR"

    def test_model_name(self):
        assert FinBERTAnalyzer.model_name == "lucas-leme/FinBERT-PT-BR"


# ── preprocess ────────────────────────────────────────────────────────────────


class TestPreprocess:
    def setup_method(self):
        # __new__ skips __init__; preprocess has no dependency on DB or model
        self.obj = FinBERTAnalyzer.__new__(FinBERTAnalyzer)

    def test_url_replaced(self):
        result = self.obj.preprocess("Veja https://t.co/abc123 sobre o IBOV")
        assert "https://" not in result
        assert "[URL]" in result

    def test_emoji_removed(self):
        result = self.obj.preprocess("🚨 Alerta de mercado")
        assert "🚨" not in result

    def test_mention_replaced(self):
        result = self.obj.preprocess("Segundo @InfoMoney o IBOV subiu")
        assert "@InfoMoney" not in result
        assert "[MENTION]" in result

    def test_hashtag_symbol_removed(self):
        result = self.obj.preprocess("Alta do #IBOV hoje")
        assert "#" not in result
        assert "IBOV" in result

    def test_extra_spaces_collapsed(self):
        result = self.obj.preprocess("texto   com   espaços")
        assert "  " not in result

    def test_lowercase_applied(self):
        result = self.obj.preprocess("Alta do Mercado Hoje")
        assert "alta" in result
        assert "mercado" in result

    def test_financial_ticker_preserved(self):
        result = self.obj.preprocess("Comprei PETR4 e VALE3 hoje")
        assert "PETR4" in result
        assert "VALE3" in result

    def test_financial_entity_preserved(self):
        result = self.obj.preprocess("Decisão do COPOM sobre a SELIC")
        assert "COPOM" in result
        assert "SELIC" in result

    def test_full_tweet(self):
        tweet = "🚨 Alta do #IBOV! Veja https://t.co/abc @InfoMoney PETR4 em alta"
        result = self.obj.preprocess(tweet)
        assert "🚨" not in result
        assert "#" not in result
        assert "https://" not in result
        assert "@InfoMoney" not in result
        assert "IBOV" in result
        assert "PETR4" in result
        assert "[URL]" in result
        assert "[MENTION]" in result


# ── normalize_label ───────────────────────────────────────────────────────────


class TestNormalizeLabel:
    def setup_method(self):
        self.obj = FinBERTAnalyzer.__new__(FinBERTAnalyzer)

    def test_positive_mapped(self):
        assert self.obj.normalize_label("POSITIVE") == "positivo"

    def test_negative_mapped(self):
        assert self.obj.normalize_label("NEGATIVE") == "negativo"

    def test_neutral_mapped(self):
        assert self.obj.normalize_label("NEUTRAL") == "neutro"

    def test_unknown_label_returns_unknown(self):
        assert self.obj.normalize_label("SURPRISE") == "unknown"


# ── predict_text ──────────────────────────────────────────────────────────────


class TestPredictText:
    def test_returns_model_output(self, analyzer, mock_model):
        result = analyzer.predict_text("Alta do IBOV hoje", batch_size=32)
        mock_model.assert_called_once_with("Alta do IBOV hoje", batch_size=32)
        assert result == [{"label": "POSITIVE", "score": 0.95}]

    def test_batch_size_forwarded(self, analyzer, mock_model):
        analyzer.predict_text("texto", batch_size=16)
        assert mock_model.call_args.kwargs["batch_size"] == 16


# ── run ───────────────────────────────────────────────────────────────────────


class TestRun:
    def _rows(self):
        return pd.DataFrame(
            {
                "id": [1, 2],
                "tweet_id": ["t1", "t2"],
                "note_tweet": [
                    "Alta do #IBOV! https://t.co/abc",
                    "@InfoMoney PETR4 subiu hoje",
                ],
            }
        )

    def test_empty_db_returns_empty_dataframe(self, analyzer):
        analyzer.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = (
            pd.DataFrame()
        )
        assert analyzer.run().empty

    def test_clear_tweets_column_added(self, analyzer):
        analyzer.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = (
            self._rows()
        )
        analyzer.db_manager.insert_tweets_classification.return_value = True
        result = analyzer.run()
        assert "clear_tweets" in result.columns

    def test_preprocess_applied(self, analyzer):
        analyzer.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = (
            self._rows()
        )
        analyzer.db_manager.insert_tweets_classification.return_value = True
        result = analyzer.run()
        assert result["clear_tweets"].str.contains("#").sum() == 0
        assert result["clear_tweets"].str.contains("https://").sum() == 0

    def test_predicted_sentiment_column_added(self, analyzer):
        analyzer.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = (
            self._rows()
        )
        analyzer.db_manager.insert_tweets_classification.return_value = True
        result = analyzer.run()
        assert "predicted_sentiment" in result.columns

    def test_save_called_once_per_row(self, analyzer):
        analyzer.db_manager.query_all_tweets_with_human_but_not_finBERT_classification.return_value = (
            self._rows()
        )
        analyzer.db_manager.insert_tweets_classification.return_value = True
        twitters_predicted = analyzer.run()
        analyzer.save(twitters_predicted)
        assert analyzer.db_manager.insert_tweets_classification.call_count == 2


# ── save ──────────────────────────────────────────────────────────────────────


class TestSave:
    def _rows(self, sentiments: list[list[dict]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": list(range(1, len(sentiments) + 1)),
                "tweet_id": [f"t{i}" for i in range(1, len(sentiments) + 1)],
                "note_tweet": ["tweet"] * len(sentiments),
                "predicted_sentiment": sentiments,
            }
        )

    def test_insert_called_for_each_prediction(self, analyzer):
        rows = self._rows(
            [
                [{"label": "POSITIVE", "score": 0.9}],
                [{"label": "NEGATIVE", "score": 0.8}],
            ]
        )
        analyzer.db_manager.insert_tweets_classification.return_value = True
        analyzer.save(rows)
        assert analyzer.db_manager.insert_tweets_classification.call_count == 2

    def test_label_normalized_before_save(self, analyzer):
        rows = self._rows([[{"label": "POSITIVE", "score": 0.95}]])
        analyzer.db_manager.insert_tweets_classification.return_value = True
        analyzer.save(rows)
        kwargs = analyzer.db_manager.insert_tweets_classification.call_args.kwargs
        assert kwargs["sentiment"] == "positivo"

    def test_classificator_is_finbert(self, analyzer):
        rows = self._rows([[{"label": "NEUTRAL", "score": 0.7}]])
        analyzer.db_manager.insert_tweets_classification.return_value = True
        analyzer.save(rows)
        kwargs = analyzer.db_manager.insert_tweets_classification.call_args.kwargs
        assert kwargs["classificator"] == "FinBERT-PT-BR"

    def test_score_passed_correctly(self, analyzer):
        rows = self._rows([[{"label": "POSITIVE", "score": 0.9512}]])
        analyzer.db_manager.insert_tweets_classification.return_value = True
        analyzer.save(rows)
        kwargs = analyzer.db_manager.insert_tweets_classification.call_args.kwargs
        assert kwargs["score"] == pytest.approx(0.9512)

    def test_tweet_id_passed_correctly(self, analyzer):
        rows = self._rows([[{"label": "NEGATIVE", "score": 0.88}]])
        analyzer.db_manager.insert_tweets_classification.return_value = True
        analyzer.save(rows)
        kwargs = analyzer.db_manager.insert_tweets_classification.call_args.kwargs
        assert kwargs["tweet_id"] == 1
