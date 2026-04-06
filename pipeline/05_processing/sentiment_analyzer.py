from transformers import pipeline as hf_pipeline
from shared.database import DatabaseManager

MODEL_NAME = "lucas-leme/FinBERT-PT-BR"
CLASSIFICATOR = "FinBERT-PT-BR"


def load_model():
    """Carrega o pipeline de análise de sentimento do FinBERT-PT-BR."""
    return hf_pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


def run(batch_size: int = 32):
    """
    Lê tweets pré-processados da tabela tweets, executa inferência
    e grava resultados em tweets_classification com
    classificator = 'FinBERT-PT-BR'.
    TODO: implementar após conclusão de 04_preprocessing.
    """
    raise NotImplementedError("Implementar após etapa 04_preprocessing.")


if __name__ == '__main__':
    run()
