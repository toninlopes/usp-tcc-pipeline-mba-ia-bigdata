import importlib.util
from pathlib import Path

from text_cleaner import (
    replace_urls,
    remove_emojis,
    replace_mentions,
    remove_hashtags,
    space_normalization,
    lowercase_normalization,
    remove_stopwords,
    lematize,
    clean,
)

import pytest


class TestURLCleaning:
    def test_http_url_replaced(self):
        assert (
            replace_urls("Veja em http://example.com o resultado")
            == "Veja em [URL] o resultado"
        )

    def test_https_url_replaced(self):
        assert replace_urls("Acesse https://t.co/abc123") == "Acesse [URL]"

    def test_multiple_urls_replaced(self):
        result = replace_urls("http://a.com e https://b.com")
        assert result == "[URL] e [URL]"

    def test_no_url_unchanged(self):
        assert replace_urls("Texto sem link") == "Texto sem link"


class TestEmojiHandling:
    def test_emoji_converted_to_name_code(self):
        result = remove_emojis("🚨 Alerta!")
        assert "🚨" not in result
        assert ":police_car_light:" in result

    def test_multiple_emojis_converted(self):
        result = remove_emojis("📈📉")
        assert "📈" not in result
        assert "📉" not in result

    def test_text_without_emoji_unchanged(self):
        assert remove_emojis("Texto normal") == "Texto normal"


class TestMentionCleaning:
    def test_mention_replaced(self):
        assert replace_mentions("Olá @InfoMoney tudo bem?") == "Olá [MENTION] tudo bem?"

    def test_multiple_mentions_replaced(self):
        result = replace_mentions("@user1 e @user2 postaram")
        assert result == "[MENTION] e [MENTION] postaram"

    def test_no_mention_unchanged(self):
        assert replace_mentions("Sem menção aqui") == "Sem menção aqui"


class TestHashtagCleaning:
    def test_hashtag_symbol_removed(self):
        assert remove_hashtags("Alta do #IBOV hoje") == "Alta do IBOV hoje"

    def test_multiple_hashtags_cleaned(self):
        assert remove_hashtags("#IBOV e #B3 em alta") == "IBOV e B3 em alta"

    def test_hashtag_text_preserved(self):
        result = remove_hashtags("#MercadoFinanceiro")
        assert "MercadoFinanceiro" in result
        assert "#" not in result


class TestWhitespaceCleaning:
    def test_multiple_spaces_collapsed(self):
        assert space_normalization("texto   com   espaços") == "texto com espaços"

    def test_leading_trailing_spaces_stripped(self):
        assert space_normalization("  texto  ") == "texto"

    def test_newlines_collapsed(self):
        assert space_normalization("linha1\n\nlinha2") == "linha1 linha2"


class TestLowercaseNormalization:
    def test_text_lowercased(self):
        assert lowercase_normalization("Alta do IBOV!") == "alta do IBOV!"

    def test_tickers_preserved(self):
        result = lowercase_normalization("Investimento em PETR4 e VALE3")
        assert result == "investimento em PETR4 e VALE3"
        assert "PETR4" in result
        assert "VALE3" in result

    def test_financial_entities_preserved(self):
        result = lowercase_normalization("Investimento em B3 e Petrobras")
        assert "B3" in result
        assert "petrobras" in result


class TestRemoveStopwords:
    def test_stopwords_removed(self):
        assert (
            remove_stopwords(
                "Este é um teste de remoção de stopwords, para verificar se funciona corretamente"
            )
            == "teste remoção stopwords, verificar funciona corretamente"
        )

    def test_no_stopwords_unchanged(self):
        assert (
            remove_stopwords("Texto contem nenhuma stopwords")
            == "Texto contem nenhuma stopwords"
        )


class TestLematization:
    def test_lemmatization(self):
        assert lematize("correr correndo correu") == "correr correr correr"

    def test_lemmatization_with_financial_terms(self):
        assert (
            lematize("🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney")
            == "🚨 Alta de o # IBOV ! saiba mais em https://t.co/abc @InfoMoney"
        )

    def test_lemmatization_with_entities(self):
        assert (
            lematize("Investimento em B3 e Petrobras está em alta")
            == "Investimento em B3 e Petrobras estar em alta"
        )

    def test_lemmatization_with_url_and_mentions(self):
        assert (
            lematize("Confira [URL] e siga [MENTION]")
            == "Confira [ URL _ e siga [ MENTION _"
        )

    def test_lemmatization_with_emojis(self):
        assert (
            lematize("Confira :finance_chart_with_upwards_trend: e :moneybag:")
            == "Confira : finance_chart_with_upwards_trend : e : moneybag :"
        )

    def test_lemmatization_with_stopwords(self):
        assert (
            lematize("Este é um teste de lematização")
            == "este ser um teste de lematização"
        )


class TestCombined:
    def test_full_tweet_sample(self):
        tweet = "🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney"
        result = clean(tweet)

        assert "🚨" not in result
        assert ": police_car_light :" in result
        assert "#" not in result
        assert "IBOV" in result
        assert "[ URL _" in result
        assert "[ MENTION _" in result
        assert "https://" not in result
        assert "@InfoMoney" not in result

        assert result == ": police_car_light : alta IBOV ! saber [ URL _ [ MENTION _"

    def test_full_tweet_with_financial_terms(self):
        tweet = "Investimento em PETR4 e VALE3 está em alta! Veja mais em https://t.co/abc @FinanceNews"
        result = clean(tweet)
        assert "PETR4" in result
        assert "VALE3" in result
        assert "[ URL _" in result
        assert "[ MENTION _" in result
        assert "https://" not in result
        assert "@FinanceNews" not in result

        assert result == "investimento PETR4 VALE3 alto ! ver [ URL _ [ MENTION _"
