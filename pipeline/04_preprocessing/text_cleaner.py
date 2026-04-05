import re
import emoji
import pandas as pd
from shared.database import DatabaseManager

db_manager = DatabaseManager()


def clean(text: str) -> str:
    """Remove ruídos típicos de tweets para preparar o texto para o FinBERT-PT-BR.
    Regras baseadas nos padrões identificados na EDA (03_eda)."""
    text = re.sub(r"http\S+", "[URL]", text)  # URLs
    text = emoji.demojize(text)  # substitui emojis pelos seus respectivos name codes
    text = re.sub(r"@\w+", "[MENTION]", text)  # menções
    text = re.sub(r"#(\w+)", r"\1", text)  # hashtags → texto limpo
    text = re.sub(r"\s+", " ", text).strip()  # espaços múltiplos
    return text


if __name__ == "__main__":
    # Teste rápido
    result = db_manager.query_all_tweets_with_human_classification()

    df = pd.DataFrame(
        result,
        columns=[
            "id",
            "tweet_id",
            "username",
            "note_tweet",
            "created_at",
            "likes",
            "hashtags",
            "tweet",
            "sentiment",
            "is_finance_tweet",
            "has_human_classification",
        ],
    )

    for tweet in df["note_tweet"].head(2):
        print("Original:", tweet)
        print(" " * 40)
        print("Limpo:", clean(tweet))
        print("-" * 40)

    sample = "🚨 Alta do #IBOV! Saiba mais em https://t.co/abc @InfoMoney"
    print(clean(sample))
