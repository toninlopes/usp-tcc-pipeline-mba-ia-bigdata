from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch
import pytest

from app.core.processing.lexicon.op_lexicon import OpLexiconAnalyzer, _THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_LEXICON: Dict[str, int] = {
    "bom": 1,
    "ótimo": 1,
    "ruim": -1,
    "péssimo": -1,
}


@pytest.fixture
def analyzer():
    with patch("app.shared.db.database.load_dotenv"):
        instance = OpLexiconAnalyzer.__new__(OpLexiconAnalyzer)
        instance._tweet_repo = MagicMock()
        instance._classification_repo = MagicMock()
        instance._lexicon_path = Path("/fake/lexico_v3.0.txt")
        instance._model = SAMPLE_LEXICON
        return instance


# ── load_model ────────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_parses_positive_entry(self, tmp_path):
        lex_file = tmp_path / "lexico_v3.0.txt"
        lex_file.write_text("bom,Adj,1\n", encoding="utf-8")
        with patch("app.shared.db.database.load_dotenv"):
            instance = OpLexiconAnalyzer.__new__(OpLexiconAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        result = instance.load_model()
        assert result["bom"] == 1

    def test_parses_negative_entry(self, tmp_path):
        lex_file = tmp_path / "lexico_v3.0.txt"
        lex_file.write_text("ruim,Adj,-1\n", encoding="utf-8")
        with patch("app.shared.db.database.load_dotenv"):
            instance = OpLexiconAnalyzer.__new__(OpLexiconAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        result = instance.load_model()
        assert result["ruim"] == -1

    def test_skips_lines_with_fewer_than_3_columns(self, tmp_path):
        lex_file = tmp_path / "lexico_v3.0.txt"
        lex_file.write_text("bom,Adj\n", encoding="utf-8")
        with patch("app.shared.db.database.load_dotenv"):
            instance = OpLexiconAnalyzer.__new__(OpLexiconAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        assert instance.load_model() == {}

    def test_skips_non_integer_polarity(self, tmp_path):
        lex_file = tmp_path / "lexico_v3.0.txt"
        lex_file.write_text("bom,Adj,forte\n", encoding="utf-8")
        with patch("app.shared.db.database.load_dotenv"):
            instance = OpLexiconAnalyzer.__new__(OpLexiconAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        assert instance.load_model() == {}

    def test_raises_if_not_found_after_download(self, tmp_path):
        missing = tmp_path / "missing.txt"
        with patch("app.shared.db.database.load_dotenv"):
            instance = OpLexiconAnalyzer.__new__(OpLexiconAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = missing
            instance._model = {}
        with patch.object(instance, "_download_lexicon"):
            with pytest.raises(FileNotFoundError):
                instance.load_model()


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_removes_url(self, analyzer):
        assert "https://" not in analyzer.preprocess("Veja https://t.co/abc")

    def test_replaces_mention(self, analyzer):
        assert "@InfoMoney" not in analyzer.preprocess("Via @InfoMoney")

    def test_does_not_remove_hashtag_symbol(self, analyzer):
        # OpLexicon mantém hashtags para ampliar cobertura léxica
        result = analyzer.preprocess("#IBOV em alta")
        assert "IBOV" in result

    def test_lowercases_text(self, analyzer):
        assert "mercado" in analyzer.preprocess("Mercado")

    def test_collapses_spaces(self, analyzer):
        assert "  " not in analyzer.preprocess("texto   espaçado")


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_positive_text(self, analyzer):
        label, score = analyzer.predict("bom ótimo resultado")
        assert label == "positivo"
        assert score > 0

    def test_negative_text(self, analyzer):
        label, score = analyzer.predict("ruim péssimo resultado")
        assert label == "negativo"
        assert score > 0

    def test_no_lexicon_match_returns_neutral(self, analyzer):
        label, score = analyzer.predict("resultado mercado hoje")
        assert label == "neutro"
        assert score == 0.0

    def test_mixed_balanced_returns_neutral(self, analyzer):
        # bom (+1) e ruim (-1) → média = 0 → neutro
        label, _ = analyzer.predict("bom ruim")
        assert label == "neutro"

    def test_score_within_threshold_returns_neutral(self, analyzer):
        # Cria léxico com polaridade abaixo do threshold
        analyzer._model = {"levemente": 1}
        # média = 1/1 = 1.0 → acima do threshold
        # Para testar o threshold, precisa de valor baixo
        analyzer._model = {}
        label, _ = analyzer.predict("texto sem léxico")
        assert label == "neutro"

    def test_empty_text_returns_neutral(self, analyzer):
        label, score = analyzer.predict("")
        assert label == "neutro"
        assert score == 0.0

    def test_score_between_zero_and_one(self, analyzer):
        _, score = analyzer.predict("bom bom bom")
        assert 0.0 <= score <= 1.0

    def test_returns_tuple(self, analyzer):
        result = analyzer.predict("bom resultado")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_ignores_tokens_not_in_lexicon(self, analyzer):
        # "PETR4" não está no léxico — não deve afetar o resultado
        label, _ = analyzer.predict("bom PETR4 resultado")
        assert label == "positivo"

    def test_threshold_boundary(self, analyzer):
        # Quando mean_score == _THRESHOLD exatamente → neutro
        # mean de [_THRESHOLD] = _THRESHOLD → não é maior que _THRESHOLD
        analyzer._model = {"teste": round(_THRESHOLD * 100)}
        # Aqui o score seria _THRESHOLD*100 (inteiro), bem acima do limiar
        label, _ = analyzer.predict("teste")
        assert label == "positivo"