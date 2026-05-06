from unittest.mock import MagicMock, patch
from typing import Any, Tuple
import pytest

from app.core.processing.bert.bert_analyzer import BertSentimentAnalyzer


# ── Implementação mínima para testar a classe abstrata ────────────────────────

class ConcreteBertAnalyzer(BertSentimentAnalyzer):
    model_name = "test/model"
    classificator = "TestBert"

    def load_model(self) -> Any:
        return MagicMock()

    def preprocess(self, text: str) -> str:
        return text

    def run(self):
        import pandas as pd
        return pd.DataFrame()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    with patch("app.shared.db.database.load_dotenv"):
        instance = ConcreteBertAnalyzer.__new__(ConcreteBertAnalyzer)
        instance._tweet_repo = MagicMock()
        instance._classification_repo = MagicMock()
        instance._model = MagicMock()
        instance._model.return_value = [{"label": "POSITIVE", "score": 0.95}]
        return instance


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_tuple(self, analyzer):
        result = analyzer.predict("Mercado em alta")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_normalized_label_and_score(self, analyzer):
        label, score = analyzer.predict("Mercado em alta")
        assert label == "positivo"
        assert score == pytest.approx(0.95)

    def test_maps_negative_label(self, analyzer):
        analyzer._model.return_value = [{"label": "NEGATIVE", "score": 0.88}]
        label, score = analyzer.predict("Queda acentuada")
        assert label == "negativo"
        assert score == pytest.approx(0.88)

    def test_maps_neutral_label(self, analyzer):
        analyzer._model.return_value = [{"label": "NEUTRAL", "score": 0.75}]
        label, _ = analyzer.predict("Volume estável")
        assert label == "neutro"

    def test_truncates_text_to_512_chars(self, analyzer):
        long_text = "a" * 600
        analyzer.predict(long_text)
        called_text = analyzer._model.call_args[0][0]
        assert len(called_text) == 512

    def test_passes_batch_size_1(self, analyzer):
        analyzer.predict("texto")
        call_kwargs = analyzer._model.call_args[1]
        assert call_kwargs.get("batch_size") == 1

    def test_short_text_not_truncated(self, analyzer):
        text = "Ibovespa sobe 1%"
        analyzer.predict(text)
        called_text = analyzer._model.call_args[0][0]
        assert called_text == text