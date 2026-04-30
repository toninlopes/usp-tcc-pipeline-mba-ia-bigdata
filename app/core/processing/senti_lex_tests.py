from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, mock_open, patch
import pytest

from app.core.processing.senti_lex import (
    SentiLexAnalyzer,
    _apply_negation,
    _confidence,
    NEGATION_WINDOW,
)


# ── _apply_negation ───────────────────────────────────────────────────────────

class TestApplyNegation:
    def test_inverts_token_after_negation(self):
        tokens = ["não", "bom", "mercado"]
        scores = [0, 1, 0]
        result = _apply_negation(tokens, scores)
        assert result[1] == -1

    def test_does_not_invert_token_outside_window(self):
        tokens = ["não", "x", "x", "x", "bom"]
        scores = [0, 0, 0, 0, 1]
        result = _apply_negation(tokens, scores)
        assert result[4] == 1  # fora da janela de 3

    def test_respects_negation_window_size(self):
        # Janela = 3: índices 1, 2, 3 são afetados; índice 4 não
        tokens = ["não", "a", "b", "c", "d"]
        scores = [0, 1, 1, 1, 1]
        result = _apply_negation(tokens, scores)
        assert result[1] == -1
        assert result[2] == -1
        assert result[3] == -1
        assert result[4] == 1

    def test_does_not_invert_zero_scores(self):
        tokens = ["não", "neutro"]
        scores = [0, 0]
        result = _apply_negation(tokens, scores)
        assert result[1] == 0

    def test_no_negation_token_unchanged(self):
        tokens = ["bom", "resultado"]
        scores = [1, 0]
        result = _apply_negation(tokens, scores)
        assert result == [1, 0]

    def test_empty_input(self):
        assert _apply_negation([], []) == []

    def test_multiple_negations(self):
        tokens = ["não", "bom", "nunca", "ruim"]
        scores = [0, 1, 0, -1]
        result = _apply_negation(tokens, scores)
        assert result[1] == -1   # negado por "não"
        assert result[3] == 1    # negado por "nunca"


# ── _confidence ───────────────────────────────────────────────────────────────

class TestConfidence:
    def test_returns_zero_for_zero_tokens(self):
        assert _confidence(5, 0) == 0.0

    def test_returns_zero_for_zero_score(self):
        assert _confidence(0, 10) == 0.0

    def test_caps_at_one(self):
        assert _confidence(100, 1) == 1.0

    def test_proportional_to_score(self):
        assert _confidence(2, 10) == pytest.approx(0.2)

    def test_uses_absolute_value(self):
        assert _confidence(-3, 10) == pytest.approx(0.3)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_LEXICON_CONTENT = (
    "bom.Adj,Flex=boa;bons;boas;POL:N0=1;POL:N1=1\n"
    "ruim.Adj,Flex=ruins;POL:N0=-1;POL:N1=-1\n"
    "neutro.Adj,Flex=neutra;POL:N0=0\n"
    "inválido_sem_pol\n"
)

SAMPLE_LEXICON: Dict[str, int] = {
    "bom": 1,
    "ruim": -1,
    "neutro": 0,
}


@pytest.fixture
def analyzer():
    with patch("app.shared.database.load_dotenv"):
        instance = SentiLexAnalyzer.__new__(SentiLexAnalyzer)
        instance._tweet_repo = MagicMock()
        instance._classification_repo = MagicMock()
        instance._lexicon_path = Path("/fake/sentiLex-PT02.txt")
        instance._model = SAMPLE_LEXICON
        return instance


# ── load_model ────────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_parses_positive_entry(self, tmp_path):
        lex_file = tmp_path / "sentiLex-PT02.txt"
        lex_file.write_text("bom.Adj,Flex=boa;POL:N0=1\n", encoding="utf-8")
        with patch("app.shared.database.load_dotenv"):
            instance = SentiLexAnalyzer.__new__(SentiLexAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        result = instance.load_model()
        assert result["bom"] == 1

    def test_parses_negative_entry(self, tmp_path):
        lex_file = tmp_path / "sentiLex-PT02.txt"
        lex_file.write_text("ruim.Adj,Flex=ruins;POL:N0=-1\n", encoding="utf-8")
        with patch("app.shared.database.load_dotenv"):
            instance = SentiLexAnalyzer.__new__(SentiLexAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        result = instance.load_model()
        assert result["ruim"] == -1

    def test_skips_malformed_lines(self, tmp_path):
        lex_file = tmp_path / "sentiLex-PT02.txt"
        lex_file.write_text("linha_invalida_sem_ponto_virgula\n", encoding="utf-8")
        with patch("app.shared.database.load_dotenv"):
            instance = SentiLexAnalyzer.__new__(SentiLexAnalyzer)
            instance._tweet_repo = MagicMock()
            instance._classification_repo = MagicMock()
            instance._lexicon_path = lex_file
            instance._model = {}
        result = instance.load_model()
        assert result == {}

    def test_raises_if_not_found_after_download(self, tmp_path):
        missing = tmp_path / "missing.txt"
        with patch("app.shared.database.load_dotenv"):
            instance = SentiLexAnalyzer.__new__(SentiLexAnalyzer)
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

    def test_removes_hashtag_symbol(self, analyzer):
        result = analyzer.preprocess("Alta do #IBOV")
        assert "#" not in result
        assert "IBOV" in result

    def test_lowercases_text(self, analyzer):
        assert "mercado" in analyzer.preprocess("Mercado")

    def test_collapses_spaces(self, analyzer):
        assert "  " not in analyzer.preprocess("texto   espaçado")


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_positive_text(self, analyzer):
        label, score = analyzer.predict("bom resultado")
        assert label == "positivo"
        assert score > 0

    def test_negative_text(self, analyzer):
        label, score = analyzer.predict("ruim resultado")
        assert label == "negativo"
        assert score > 0

    def test_neutral_text_no_lexicon_match(self, analyzer):
        label, score = analyzer.predict("resultado mercado hoje")
        assert label == "neutro"

    def test_negation_inverts_positive(self, analyzer):
        # "não bom" deve resultar em negativo
        label, _ = analyzer.predict("não bom")
        assert label == "negativo"

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