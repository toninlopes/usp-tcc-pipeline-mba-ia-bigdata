from pathlib import Path
import importlib.util
from unittest import result

import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline
from shared.database import DatabaseManager


def _load_clean():
    cleaner_path = (
        Path(__file__).resolve().parents[1] / "04_preprocessing" / "text_cleaner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipeline_04_preprocessing_text_cleaner",
        cleaner_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load cleaner module from {cleaner_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean


clean = _load_clean()
db_manager = DatabaseManager()

MODEL_NAME = "lucas-leme/FinBERT-PT-BR"
CLASSIFICATOR = "FinBERT-PT-BR"


def load_model_as_pipeline():
    """Carrega o pipeline de análise de sentimento do FinBERT-PT-BR."""
    finbert_pt_br_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    finbert_pt_br_model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

    return pipeline(
        task="text-classification",
        model=finbert_pt_br_model,
        tokenizer=finbert_pt_br_tokenizer,
    )


def run(batch_size: int = 32):
    """
    Lê tweets pré-processados da tabela tweets, executa inferência
    e grava resultados em tweets_classification com
    classificator = 'FinBERT-PT-BR'.
    """

    results = db_manager.query_all_tweets_with_human_but_not_finBERT_classification()

    if not len(results):
        print("No tweets found for classification.")
        return

    results["clear_tweets"] = results["note_tweet"].apply(lambda x: clean(x))

    prediction_pipeline = load_model_as_pipeline()

    results["predicted_sentiment"] = results["clear_tweets"].apply(
        lambda x: prediction_pipeline(x, batch_size=batch_size),
    )

    for record in results.itertuples():
        for sentiment_info in record.predicted_sentiment:
            sentiment_label = sentiment_info["label"]
            score = sentiment_info["score"]

            converted_label = {
                "NEGATIVE": "negativo",
                "NEUTRAL": "neutro",
                "POSITIVE": "positivo",
            }.get(sentiment_label, "unknown")

            result = db_manager.insert_tweets_classification(
                tweet_id=record.id,
                is_finance_news=1,
                why_is_finance_news="",
                sentiment=converted_label,
                why_sentiment="",
                classificator=CLASSIFICATOR,
                score=score,
            )

            if result:
                print(
                    f"Classification saved for Tweet ID: {record.tweet_id} | Sentiment: {sentiment_label} | Score: {score:.4f}",
                )
            else:
                print(
                    f"Failed to save classification for Tweet ID: {record.tweet_id}",
                )


if __name__ == "__main__":
    run()
