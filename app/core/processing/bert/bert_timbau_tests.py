from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import sys

# Bloqueia HuggingFace e torch antes de qualquer import do módulo
for _mod in [
    "transformers",
    "transformers.pipelines",
    "transformers.models",
    "transformers.models.auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
    "torch",
]:
    sys.modules.setdefault(_mod, MagicMock())

from app.core.processing.bert.bert_timbau import BERTimbauAnalyzer, _FINE_TUNED_PATH


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.return_value = [{"label": "positivo", "score": 0.91}]
    return model


@pytest.fixture
def analyzer(mock_model, tmp_path):
    """BERTimbauAnalyzer com DB e modelo mockados — ignora __init__."""
    instance = BERTimbauAnalyzer.__new__(BERTimbauAnalyzer)
    instance._tweet_repo = MagicMock()
    instance._classification_repo = MagicMock()
    instance._model = mock_model
    instance.load_model = MagicMock(return_value=mock_model)
    return instance


@pytest.fixture
def tweets_df():
    return pd.DataFrame([
        {
            "id": 1,
            "tweet_id": "abc",
            "note_tweet": "PETR4 em alta após resultado recorde",
            "sentiment": "positivo",
            "is_finance_tweet": 1,
            "has_human_classification": True,
            "has_finbert_classification": False,
        },
        {
            "id": 2,
            "tweet_id": "def",
            "note_tweet": "IBOV cai 2% com incerteza fiscal",
            "sentiment": "negativo",
            "is_finance_tweet": 1,
            "has_human_classification": True,
            "has_finbert_classification": False,
        },
    ])


# ── Atributos de classe ───────────────────────────────────────────────────────

class TestClassAttributes:
    def test_classificator(self):
        assert BERTimbauAnalyzer.classificator == "BERTimbau"

    def test_model_name_points_to_fine_tuned_path(self):
        assert "bert-timbau-sentiment" in BERTimbauAnalyzer.model_name


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_raises_when_model_not_found(self):
        with patch("app.core.processing.bert.bert_timbau._FINE_TUNED_PATH") as mock_path:
            mock_path.exists.return_value = False
            with pytest.raises(RuntimeError, match="Modelo fine-tuned não encontrado"):
                BERTimbauAnalyzer()

    def test_does_not_raise_when_model_exists(self, tmp_path):
        fake_model_dir = tmp_path / "bert-timbau-sentiment"
        fake_model_dir.mkdir()
        with patch("app.core.processing.bert.bert_timbau._FINE_TUNED_PATH", fake_model_dir), \
             patch("app.shared.database.load_dotenv"), \
             patch.object(BERTimbauAnalyzer, "load_model", return_value=MagicMock()):
            instance = BERTimbauAnalyzer()
        assert instance is not None


# ── normalize_label ───────────────────────────────────────────────────────────

class TestNormalizeLabel:
    def test_portuguese_positive_passes_through(self, analyzer):
        assert analyzer.normalize_label("positivo") == "positivo"

    def test_portuguese_negative_passes_through(self, analyzer):
        assert analyzer.normalize_label("negativo") == "negativo"

    def test_portuguese_neutral_passes_through(self, analyzer):
        assert analyzer.normalize_label("neutro") == "neutro"

    def test_english_positive_falls_back_to_base(self, analyzer):
        assert analyzer.normalize_label("POSITIVE") == "positivo"

    def test_english_negative_falls_back_to_base(self, analyzer):
        assert analyzer.normalize_label("NEGATIVE") == "negativo"

    def test_english_neutral_falls_back_to_base(self, analyzer):
        assert analyzer.normalize_label("NEUTRAL") == "neutro"

    def test_unknown_label_returns_unknown(self, analyzer):
        assert analyzer.normalize_label("OUTROS") == "unknown"


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_removes_url(self, analyzer):
        assert "https://" not in analyzer.preprocess("Veja https://t.co/abc")

    def test_replaces_mention(self, analyzer):
        assert "@InfoMoney" not in analyzer.preprocess("Via @InfoMoney")

    def test_removes_hashtag_symbol(self, analyzer):
        result = analyzer.preprocess("Alta do #IBOV")
        assert "#" not in result
        assert "IBOV" in result

    def test_does_not_lowercase(self, analyzer):
        """BERTimbau é cased — capitalização deve ser preservada."""
        result = analyzer.preprocess("PETR4 em Alta")
        assert "PETR4" in result
        assert "Alta" in result

    def test_preserves_ticker_casing(self, analyzer):
        result = analyzer.preprocess("Comprei VALE3 e PETR4")
        assert "VALE3" in result
        assert "PETR4" in result

    def test_collapses_spaces(self, analyzer):
        assert "  " not in analyzer.preprocess("texto   espaçado")


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_tuple(self, analyzer):
        result = analyzer.predict("PETR4 em alta")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_portuguese_label(self, analyzer):
        label, _ = analyzer.predict("PETR4 em alta")
        assert label in {"positivo", "negativo", "neutro"}

    def test_returns_score_between_zero_and_one(self, analyzer):
        _, score = analyzer.predict("PETR4 em alta")
        assert 0.0 <= score <= 1.0

    def test_maps_portuguese_label_directly(self, analyzer, mock_model):
        mock_model.return_value = [{"label": "negativo", "score": 0.85}]
        label, score = analyzer.predict("queda acentuada")
        assert label == "negativo"
        assert score == pytest.approx(0.85)

    def test_maps_english_label_via_fallback(self, analyzer, mock_model):
        mock_model.return_value = [{"label": "POSITIVE", "score": 0.77}]
        label, _ = analyzer.predict("mercado em alta")
        assert label == "positivo"

    def test_truncates_text_to_512_chars(self, analyzer):
        long_text = "a" * 600
        analyzer.predict(long_text)
        called_text = analyzer._model.call_args[0][0]
        assert len(called_text) == 512

    def test_uses_batch_size_32(self, analyzer):
        analyzer.predict("texto")
        call_kwargs = analyzer._model.call_args[1]
        assert call_kwargs.get("batch_size") == 32


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

    def test_predicted_label_is_in_portuguese(self, analyzer, tweets_df):
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df
        result = analyzer.run()
        for _, row in result.iterrows():
            label = row["predicted_sentiment"][0]["label"]
            assert label in {"positivo", "negativo", "neutro"}

    def test_preserves_casing_in_clear_tweets(self, analyzer, tweets_df):
        """Confirma que preprocess não aplica lowercase no BERTimbau."""
        analyzer._tweet_repo.query_all_tweets_with_human_but_not_finbert_classification.return_value = tweets_df
        result = analyzer.run()
        assert "PETR4" in result.iloc[0]["clear_tweets"]