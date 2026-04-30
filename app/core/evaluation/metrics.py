from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.shared.db_classification import ClassificationRepository

LABELS = ["positivo", "neutro", "negativo"]

AVAILABLE_CLASSIFICATORS = [
    "FinBERT-PT-BR",
    "SentiLex-PT",
    "OpLexicon",
]


# ── Acesso ao banco ───────────────────────────────────────────────────────────

def load_pairs(
    classificator: str,
    repo: ClassificationRepository = None,
) -> Tuple[List[str], List[str]]:
    """Carrega pares (gold standard humano, predição do modelo) do banco.

    Args:
        classificator: Nome do classificador a avaliar.
        repo: Instância do repositório. Criada automaticamente se não fornecida.

    Returns:
        Tupla (y_true, y_pred) com listas de labels em português.

    Raises:
        ValueError: Se não houver pares suficientes para avaliação.
    """
    repo = repo or ClassificationRepository()
    pairs = repo.query_classification_pairs(classificator)

    if pairs.empty:
        raise ValueError(
            f"Nenhum par encontrado para o classificador '{classificator}'.\n"
            "Execute o processamento antes de avaliar."
        )

    y_true = pairs["human_label"].tolist()
    y_pred = pairs["model_label"].tolist()
    return y_true, y_pred


# ── Métricas puras ────────────────────────────────────────────────────────────

def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Calcula a acurácia entre os rótulos verdadeiros e preditos.

    Returns:
        Acurácia em [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_f1(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, float]:
    """Calcula F1-score macro e weighted.

    Returns:
        Dicionário com chaves 'macro' e 'weighted'.
    """
    return {
        "macro": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
) -> np.ndarray:
    """Calcula a matriz de confusão com ordem de labels fixada em LABELS.

    Returns:
        Array numpy de shape (3, 3) na ordem [positivo, neutro, negativo].
    """
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def compute_report(y_true: List[str], y_pred: List[str]) -> str:
    """Gera o relatório completo de classificação por classe.

    Returns:
        String formatada com precisão, recall, F1 e suporte por classe.
    """
    return classification_report(y_true, y_pred, labels=LABELS, zero_division=0)


# ── Orquestrador ──────────────────────────────────────────────────────────────

def evaluate(
    classificator: str,
    repo: ClassificationRepository = None,
) -> Dict:
    """Avalia um classificador contra o gold standard humano.

    Computa acurácia, F1 macro, F1 weighted, matriz de confusão e
    relatório completo de classificação.

    Args:
        classificator: Nome do classificador a avaliar.
                       Ex: 'FinBERT-PT-BR', 'SentiLex-PT', 'OpLexicon'.
        repo: Repositório de classificações. Criado automaticamente se não fornecido.

    Returns:
        Dicionário com as chaves:
            - classificator: str
            - n_samples: int
            - accuracy: float
            - f1_macro: float
            - f1_weighted: float
            - confusion_matrix: np.ndarray
            - report: str
            - y_true: List[str]
            - y_pred: List[str]
    """
    y_true, y_pred = load_pairs(classificator, repo)
    f1 = compute_f1(y_true, y_pred)

    results = {
        "classificator": classificator,
        "n_samples": len(y_true),
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1_macro": f1["macro"],
        "f1_weighted": f1["weighted"],
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "report": compute_report(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return results


def print_evaluation(results: Dict) -> None:
    """Imprime o resultado de evaluate() de forma legível."""
    print(f"\n{'=' * 50}")
    print(f"Avaliação: {results['classificator']}")
    print(f"Amostras : {results['n_samples']}")
    print(f"{'=' * 50}")
    print(f"Acurácia      : {results['accuracy']:.4f}")
    print(f"F1 Macro      : {results['f1_macro']:.4f}")
    print(f"F1 Weighted   : {results['f1_weighted']:.4f}")
    print(f"\nMatriz de Confusão (ordem: {LABELS}):")
    print(results["confusion_matrix"])
    print(f"\n{results['report']}")


if __name__ == "__main__":
    import sys

    classificator = sys.argv[1] if len(sys.argv) > 1 else "FinBERT-PT-BR"
    results = evaluate(classificator)
    print_evaluation(results)