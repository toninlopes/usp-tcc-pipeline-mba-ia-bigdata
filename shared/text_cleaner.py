import re

import emoji
import nltk
import spacy

nltk.download("stopwords", quiet=True)

FINANCIAL_ENTITIES: frozenset[str] = frozenset(
    {
        # Bancos e corretoras
        "BB", "BBI", "BNDES", "BRB", "BTG", "IRB", "UBS", "XP",
        # Órgãos reguladores e fiscalizadores
        "ANATEL", "BACEN", "BCB", "CADE", "CVM", "IBGE", "TCU",
        # Infraestrutura e política monetária
        "B3", "COPOM", "SELIC",
        # Índices e instrumentos financeiros
        "BDR", "CDI", "CRI", "DY", "EPS", "FI", "FII",
        "IBOV", "IBOVESPA", "IFIX", "IPCA", "IPO",
        "JCP", "OPA", "PLR", "PN", "ROE",
    }
)


def replace_urls(text: str) -> str:
    return re.sub(r"http\S+", "[URL]", text)


def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, "")


def replace_emojis_with_codes(text: str) -> str:
    return emoji.demojize(text)


def replace_mentions(text: str) -> str:
    return re.sub(r"@\w+", "[MENTION]", text)


def remove_hashtags(text: str) -> str:
    return text.replace("#", "")


def space_normalization(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lowercase_normalization(text: str) -> str:
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
    stopwords = nltk.corpus.stopwords.words("portuguese")
    words = text.split()
    return " ".join(word for word in words if word.lower() not in stopwords)


def find_emoji_codes(text: str) -> list[str]:
    return re.findall(r":\s*\w+\s*:", text)


def lematize(text: str) -> str:
    nlp = spacy.load("pt_core_news_lg")
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


def clean(text: str) -> str:
    text = replace_urls(text)
    text = replace_emojis_with_codes(text)
    text = replace_mentions(text)
    text = remove_hashtags(text)
    text = space_normalization(text)
    text = lowercase_normalization(text)
    text = remove_stopwords(text)
    text = lematize(text)
    return text
