import pytest

from app.shared.text_cleaner import (
    replace_urls,
    remove_emojis,
    replace_emojis_with_codes,
    replace_mentions,
    remove_hashtags,
    strip_boundary_punctuation,
    space_normalization,
    lowercase_normalization,
    remove_stopwords,
    lemmatize,
    clean,
)


class TestURLCleaning:
    def test_http_url_replaced(self):
        assert replace_urls("Veja em http://example.com o resultado") == "Veja em [URL] o resultado"

    def test_https_url_replaced(self):
        assert replace_urls("Acesse https://t.co/abc123") == "Acesse [URL]"

    def test_multiple_urls_replaced(self):
        assert replace_urls("http://a.com e https://b.com") == "[URL] e [URL]"

    def test_no_url_unchanged(self):
        assert replace_urls("Texto sem link") == "Texto sem link"


class TestEmojiHandling:
    def test_emoji_converted_to_name_code(self):
        result = replace_emojis_with_codes("🚨 Alerta!")
        assert "🚨" not in result
        assert ":sirene:" in result

    def test_multiple_emojis_converted(self):
        result = replace_emojis_with_codes("📈📉")
        assert ":gráfico_subindo:" in result
        assert ":gráfico_caindo:" in result

    def test_text_without_emoji_unchanged(self):
        assert replace_emojis_with_codes("Texto normal") == "Texto normal"

    def test_removed_emoji(self):
        result = remove_emojis("🚨 Alerta!")
        assert "🚨" not in result
        assert ":police_car_light:" not in result

        result = remove_emojis("📈📉")
        assert "📈" not in result
        assert "📉" not in result


class TestMentionCleaning:
    def test_mention_replaced(self):
        assert replace_mentions("Olá @InfoMoney tudo bem?") == "Olá [MENTION] tudo bem?"

    def test_multiple_mentions_replaced(self):
        assert replace_mentions("@user1 e @user2 postaram") == "[MENTION] e [MENTION] postaram"

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


class TestStripBoundaryPunctuation:
    def test_trailing_period(self):
        assert strip_boundary_punctuation("bom.") == "bom"

    def test_trailing_comma(self):
        assert strip_boundary_punctuation("resultado,") == "resultado"

    def test_trailing_exclamation(self):
        assert strip_boundary_punctuation("alta!") == "alta"

    def test_trailing_question_mark(self):
        assert strip_boundary_punctuation("queda?") == "queda"

    def test_multiple_trailing_boundary_chars(self):
        assert strip_boundary_punctuation("bom...") == "bom"

    def test_leading_boundary_char(self):
        assert strip_boundary_punctuation("!bom") == "bom"

    def test_double_quoted_word(self):
        assert strip_boundary_punctuation('"análise"') == "análise"

    def test_single_quoted_word(self):
        assert strip_boundary_punctuation("'previsão'") == "previsão"

    def test_preserves_internal_hyphen(self):
        assert strip_boundary_punctuation("à-vontade") == "à-vontade"

    def test_preserves_text_emoticon(self):
        assert strip_boundary_punctuation(":)") == ":)"

    def test_strips_trailing_from_hashtag(self):
        assert strip_boundary_punctuation("#boa!") == "#boa"

    def test_strips_trailing_from_emoji_code(self):
        result = strip_boundary_punctuation(":chart_increasing:,")
        assert result == ":chart_increasing:"

    def test_full_sentence(self):
        result = strip_boundary_punctuation("Não gostei do resultado, mas o atendimento foi bom.")
        assert result == "Não gostei do resultado mas o atendimento foi bom"

    def test_pipeline_tokens_preserved(self):
        result = strip_boundary_punctuation("[URL] é confiável.")
        assert "[URL]" in result
        assert "." not in result

    def test_pure_punctuation_token_removed(self):
        assert strip_boundary_punctuation("bom . ruim") == "bom ruim"

    def test_no_boundary_punctuation_unchanged(self):
        assert strip_boundary_punctuation("texto simples") == "texto simples"

    def test_empty_string(self):
        assert strip_boundary_punctuation("") == ""


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
        assert "PETR4" in result
        assert "VALE3" in result

    def test_financial_entities_preserved(self):
        result = lowercase_normalization("Investimento em B3 e Petrobras")
        assert "B3" in result
        assert "petrobras" in result


class TestRemoveStopwords:
    def test_stopwords_removed(self):
        assert (
            remove_stopwords("Este é um teste de remoção de stopwords, para verificar se funciona corretamente")
            == "teste remoção stopwords, verificar funciona corretamente"
        )

    def test_no_stopwords_unchanged(self):
        assert remove_stopwords("Texto contem nenhuma stopwords") == "Texto contem nenhuma stopwords"


class TestLematization:
    def test_lemmatization(self):
        assert lemmatize("correr correndo correu") == "correr correr correr"

    def test_lemmatization_with_financial_terms(self):
        assert (
            lemmatize("🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney")
            == "🚨 Alta de o # IBOV ! saiba mais em https://t.co/abc @InfoMoney"
        )

    def test_lemmatization_with_entities(self):
        assert (
            lemmatize("Investimento em B3 e Petrobras está em alta")
            == "Investimento em B3 e Petrobras estar em alta"
        )

    def test_lemmatization_with_url_and_mentions(self):
        assert lemmatize("Confira [URL] e siga [MENTION]") == "Confira [ URL _ e siga [ MENTION _"

    def test_lemmatization_with_emojis(self):
        assert (
            lemmatize("Confira :finance_chart_with_upwards_trend: e :moneybag:")
            == "Confira : finance_chart_with_upwards_trend : e : moneybag :"
        )

    def test_lemmatization_with_stopwords(self):
        assert lemmatize("Este é um teste de lematização") == "este ser um teste de lematização"


class TestCombined:
    def test_full_tweet_sample(self):
        tweet = "🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney"
        result = clean(tweet)
        assert "🚨" not in result
        assert ": sirene :" in result
        assert "#" not in result
        assert "IBOV" in result
        assert "[ URL _" in result
        assert "[ MENTION _" in result
        assert "https://" not in result
        assert "@InfoMoney" not in result
        assert result == ": sirene : alta IBOV ! saber [ URL _ [ MENTION _"

    def test_full_tweet_with_financial_terms(self):
        tweet = "Investimento em PETR4 e VALE3 está em alta! Veja mais em https://t.co/abc @FinanceNews"
        result = clean(tweet)
        assert "PETR4" in result
        assert "VALE3" in result
        assert "[ URL _" in result
        assert "[ MENTION _" in result
        assert result == "investimento PETR4 VALE3 alto ! ver [ URL _ [ MENTION _"