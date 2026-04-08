from operator import index
import re
import emoji
from numpy import doc
import nltk
import spacy
import pandas as pd
from shared.database import DatabaseManager

db_manager = DatabaseManager()
nltk.download("stopwords")

# Siglas de entidades e instituições do mercado financeiro brasileiro
# que devem ser preservadas em maiúsculas durante a normalização textual.
FINANCIAL_ENTITIES: frozenset[str] = frozenset(
    {
        # Bancos e corretoras
        "BB",  # Banco do Brasil
        "BBI",  # BTG Pactual BBI
        "BNDES",  # Banco Nacional de Desenvolvimento Econômico e Social
        "BRB",  # Banco de Brasília
        "BTG",  # BTG Pactual
        "IRB",  # IRB Brasil RE
        "UBS",  # UBS Investment Bank
        "XP",  # XP Investimentos
        # Órgãos reguladores e fiscalizadores
        "ANATEL",  # Agência Nacional de Telecomunicações
        "BACEN",  # Banco Central do Brasil (alternativo)
        "BCB",  # Banco Central do Brasil
        "CADE",  # Conselho Administrativo de Defesa Econômica
        "CVM",  # Comissão de Valores Mobiliários
        "IBGE",  # Instituto Brasileiro de Geografia e Estatística
        "TCU",  # Tribunal de Contas da União
        # Infraestrutura e política monetária
        "B3",  # Bolsa de Valores do Brasil
        "COPOM",  # Comitê de Política Monetária
        "SELIC",  # Sistema Especial de Liquidação e de Custódia
        # Índices e instrumentos financeiros
        "BDR",  # Brazilian Depositary Receipt
        "CDI",  # Certificado de Depósito Interbancário
        "CRI",  # Certificado de Recebíveis Imobiliários
        "DY",  # Dividend Yield
        "EPS",  # Earnings Per Share
        "FI",  # Fundo de Investimento
        "FII",  # Fundo de Investimento Imobiliário
        "IBOV",  # Índice Bovespa
        "IBOVESPA",  # Índice Bovespa (forma extensa)
        "IFIX",  # Índice de Fundos Imobiliários
        "IPCA",  # Índice de Preços ao Consumidor Amplo
        "IPO",  # Initial Public Offering
        "JCP",  # Juros sobre Capital Próprio
        "OPA",  # Oferta Pública de Aquisição
        "PLR",  # Participação nos Lucros e Resultados
        "PN",  # Ações Preferenciais
        "ROE",  # Return on Equity
    }
)


def replace_urls(text: str) -> str:
    """Substitui URLs por um placeholder."""
    return re.sub(r"http\S+", "[URL]", text)


def remove_emojis(text: str) -> str:
    """Substitui emojis pelos seus respectivos name codes."""
    return emoji.demojize(text)


def replace_mentions(text: str) -> str:
    """Substitui menções por um placeholder."""
    return re.sub(r"@\w+", "[MENTION]", text)


def remove_hashtags(text: str) -> str:
    """Remove hashtags do texto."""
    return text.replace("#", "")


def space_normalization(text: str) -> str:
    """Remove espaços extras do texto."""
    return re.sub(r"\s+", " ", text).strip()


def lowercase_normalization(text: str) -> str:
    """Converte o texto para minúsculas, preservando tickers (ex: PETR4) e entidades financeiras."""
    words = text.split()
    result = []
    for word in words:
        clean_word = re.sub(r"[^\w]", "", word).upper()
        is_mention = word.startswith("[MENTION]")
        is_url = word.startswith("[URL]")
        is_ticker = bool(re.fullmatch(r"[A-Z]{4,5}[0-9]{1,2}", clean_word))
        is_entity = clean_word in FINANCIAL_ENTITIES
        result.append(
            word if (is_ticker or is_entity or is_mention or is_url) else word.lower()
        )
    return " ".join(result)


def remove_stopwords(text: str) -> str:
    """Remove stopwords comuns do português para reduzir ruído no texto."""
    stopwords = nltk.corpus.stopwords.words("portuguese")
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)


def find_emoji_codes(text: str) -> list[str]:
    """Lista todos os emoji codes (ex: :fire:) presentes no texto."""
    return re.findall(r":\s*\w+\s*:", text)


def lematize(text: str) -> str:
    """Lematiza o texto para reduzir palavras à sua forma base."""
    npl = spacy.load("pt_core_news_lg")
    doc = npl(text)
    lemmatized_words = " ".join(token.lemma_ for token in doc)
    # lemmatized_words_with_url = re.sub(r"\[ URL _", "[URL]", lemmatized_words)
    # lemmatized_words_with_mention = re.sub(
    #     r"\[ MENTION _", "[MENTION]", lemmatized_words_with_url
    # )
    # emoji_codes = find_emoji_codes(lemmatized_words_with_mention)
    # for code in emoji_codes:
    #     lemmatized_words_with_emojis = lemmatized_words_with_mention.replace(
    #         code, re.sub(r"\s+", " ", code).strip()
    #     )

    # print(f"Emoji codes found: {emoji_codes}")
    # print(f"Lematized: {lemmatized_words_with_emojis}")

    return lemmatized_words


def clean(text: str) -> str:
    """Remove ruídos típicos de tweets para preparar o texto para o FinBERT-PT-BR.
    Regras baseadas nos padrões identificados na EDA (03_eda)."""
    text = replace_urls(text)
    text = remove_emojis(text)
    text = replace_mentions(text)
    text = remove_hashtags(text)
    text = space_normalization(text)
    text = lowercase_normalization(text)
    text = remove_stopwords(text)
    text = lematize(text)
    return text


if __name__ == "__main__":
    # Teste rápido
    # result = db_manager.query_all_tweets_with_human_classification()

    # df = pd.DataFrame(
    #     result,
    #     columns=[
    #         "id",
    #         "tweet_id",
    #         "username",
    #         "note_tweet",
    #         "created_at",
    #         "likes",
    #         "hashtags",
    #         "tweet",
    #         "sentiment",
    #         "is_finance_tweet",
    #         "has_human_classification",
    #     ],
    # )

    # for tweet in df["note_tweet"].head(2):
    #     print("Original:", tweet)
    #     print(" " * 40)
    #     print("Limpo:", clean(tweet))
    #     print("-" * 40)

    # sample = "🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney"
    sample = "Investimento em PETR4 e VALE3 está em alta! Veja mais em https://t.co/abc @FinanceNews"
    # sample = "Confira [URL] e siga [MENTION]. :finance_chart_with_upwards_trend: #Investimentos"
    print(clean(sample))
