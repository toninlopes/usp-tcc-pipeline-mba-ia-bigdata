from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from shared.database import DatabaseManager

LABELS = ["positivo", "neutro", "negativo"]


def load_pairs():
    """
    Busca no banco os pares de classificações Humano vs FinBERT-PT-BR
    para os mesmos tweet_id.
    Retorna (y_true, y_pred).
    TODO: implementar query após etapa 05_processing.
    """
    raise NotImplementedError


def evaluate():
    y_true, y_pred = load_pairs()
    print("=== Relatório de Avaliação ===")
    print(classification_report(y_true, y_pred, labels=LABELS))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))


if __name__ == '__main__':
    evaluate()
