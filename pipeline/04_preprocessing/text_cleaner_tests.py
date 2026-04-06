import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "text_cleaner",
    Path(__file__).parent / "text_cleaner.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[union-attr]
clean = _module.clean


class TestURLCleaning:
    def test_http_url_replaced(self):
        assert clean("Veja em http://example.com o resultado") == "Veja em [URL] o resultado"

    def test_https_url_replaced(self):
        assert clean("Acesse https://t.co/abc123") == "Acesse [URL]"

    def test_multiple_urls_replaced(self):
        result = clean("http://a.com e https://b.com")
        assert result == "[URL] e [URL]"

    def test_no_url_unchanged(self):
        assert clean("Texto sem link") == "Texto sem link"


class TestEmojiHandling:
    def test_emoji_converted_to_name_code(self):
        result = clean("🚨 Alerta!")
        assert "🚨" not in result
        assert ":police_car_light:" in result

    def test_multiple_emojis_converted(self):
        result = clean("📈📉")
        assert "📈" not in result
        assert "📉" not in result

    def test_text_without_emoji_unchanged(self):
        assert clean("Texto normal") == "Texto normal"


class TestMentionCleaning:
    def test_mention_replaced(self):
        assert clean("Olá @InfoMoney tudo bem?") == "Olá [MENTION] tudo bem?"

    def test_multiple_mentions_replaced(self):
        result = clean("@user1 e @user2 postaram")
        assert result == "[MENTION] e [MENTION] postaram"

    def test_no_mention_unchanged(self):
        assert clean("Sem menção aqui") == "Sem menção aqui"


class TestHashtagCleaning:
    def test_hashtag_symbol_removed(self):
        assert clean("Alta do #IBOV hoje") == "Alta do IBOV hoje"

    def test_multiple_hashtags_cleaned(self):
        assert clean("#IBOV e #B3 em alta") == "IBOV e B3 em alta"

    def test_hashtag_text_preserved(self):
        result = clean("#MercadoFinanceiro")
        assert "MercadoFinanceiro" in result
        assert "#" not in result


class TestWhitespaceCleaning:
    def test_multiple_spaces_collapsed(self):
        assert clean("texto   com   espaços") == "texto com espaços"

    def test_leading_trailing_spaces_stripped(self):
        assert clean("  texto  ") == "texto"

    def test_newlines_collapsed(self):
        assert clean("linha1\n\nlinha2") == "linha1 linha2"


class TestCombined:
    def test_full_tweet_sample(self):
        tweet = "🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney"
        result = clean(tweet)
        assert "🚨" not in result
        assert ":police_car_light:" in result
        assert "IBOV" in result
        assert "#" not in result
        assert "[URL]" in result
        assert "[MENTION]" in result
        assert "https://" not in result
        assert "@InfoMoney" not in result

    def test_empty_string(self):
        assert clean("") == ""

    def test_only_whitespace(self):
        assert clean("   ") == ""
