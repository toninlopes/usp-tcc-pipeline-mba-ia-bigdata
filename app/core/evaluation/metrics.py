import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from app.shared.db.classification import ClassificationRepository

LABELS = ["positivo", "neutro", "negativo"]

AVAILABLE_CLASSIFICATORS = [
    "FinBERT-PT-BR",
    "SentiLex-PT",
    "OpLexicon",
    "BERTimbau",
]

BERTIMBAU_EVAL_DIR = "models/bert-timbau-sentiment/eval"

# ── Acesso ao banco ───────────────────────────────────────────────────────────

def load_pairs(
    classificator: str,
    repo: Optional[ClassificationRepository] = None,
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

# ── Leitura de artefatos do BERTimbau ────────────────────────────────────────
 
def load_pairs_from_confusion_matrix(
    cm_path: str,
) -> Tuple[List[str], List[str]]:
    """Reconstrói y_true e y_pred a partir de uma matriz de confusão em CSV.
 
    O arquivo CSV gerado pelo fine-tuner tem o formato:
        linhas  → true_<classe>
        colunas → pred_<classe>
 
    A reconstrução é lossless para todas as métricas de classificação:
    acurácia, precisão, revocação, F1 e matriz de confusão reproduzem
    exatamente os valores originais.
 
    Args:
        cm_path: Caminho para o arquivo confusion_matrix.csv.
 
    Returns:
        Tupla (y_true, y_pred) com listas de labels em português.
 
    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o formato do CSV for inesperado.
    """
    if not os.path.isfile(cm_path):
        raise FileNotFoundError(
            f"Matriz de confusão não encontrada: {cm_path}\n"
            "Execute 'make finetune' antes de avaliar o BERTimbau."
        )
 
    cm_df = pd.read_csv(cm_path, index_col=0)
 
    # Mapeia os nomes compostos do CSV ("true_negativo") para labels simples
    row_map = {f"true_{lbl}": lbl for lbl in LABELS}
    col_map = {f"pred_{lbl}": lbl for lbl in LABELS}
 
    missing_rows = set(cm_df.index) - set(row_map)
    missing_cols = set(cm_df.columns) - set(col_map)
    if missing_rows or missing_cols:
        raise ValueError(
            f"Formato inesperado na matriz de confusão.\n"
            f"Linhas desconhecidas: {missing_rows}\n"
            f"Colunas desconhecidas: {missing_cols}"
        )
 
    y_true: List[str] = []
    y_pred: List[str] = []
 
    for row_name in cm_df.index:
        true_label = row_map[row_name]
        for col_name in cm_df.columns:
            pred_label = col_map[col_name]
            count = int(cm_df.loc[row_name, col_name])
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
 
    return y_true, y_pred
 
 
def evaluate_bertimbau_from_files(
    eval_dir: str = BERTIMBAU_EVAL_DIR,
) -> Dict:
    """Avalia o BERTimbau lendo os artefatos persistidos pelo fine-tuner.
 
    Usa a matriz de confusão do conjunto de teste para reconstruir os vetores
    y_true/y_pred e calcular todas as métricas via sklearn, garantindo
    consistência com a avaliação dos demais classificadores.
 
    Args:
        eval_dir: Diretório onde estão confusion_matrix.csv e
                  classification_report.txt. Padrão: BERTIMBAU_EVAL_DIR.
 
    Returns:
        Dicionário com as mesmas chaves retornadas por evaluate():
        classificator, n_samples, accuracy, f1_macro, f1_weighted,
        confusion_matrix, report, y_true, y_pred.
 
    Raises:
        FileNotFoundError: Se confusion_matrix.csv não existir no diretório.
    """
    cm_path = os.path.join(eval_dir, "confusion_matrix.csv")
    y_true, y_pred = load_pairs_from_confusion_matrix(cm_path)
 
    f1 = compute_f1(y_true, y_pred)
 
    return {
        "classificator": "BERTimbau",
        "n_samples": len(y_true),
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1_macro": f1["macro"],
        "f1_weighted": f1["weighted"],
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
        "report": compute_report(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }

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
    repo: Optional[ClassificationRepository] = None,
) -> Dict:
    """Avalia um classificador contra o gold standard humano.
 
    Despacha automaticamente para a leitura de artefatos em disco quando
    ``classificator == "BERTimbau"``, e para o banco de dados nos demais casos.
 
    Computa acurácia, F1 macro, F1 weighted, matriz de confusão e
    relatório completo de classificação.
 
    Args:
        classificator: Nome do classificador a avaliar.
            Valores válidos: AVAILABLE_CLASSIFICATORS.
        repo: Repositório de classificações. Criado automaticamente se não
            fornecido.

    Returns:
        Dicionário com as chaves:
            - classificator: str
            - n_samples: int
            - accuracy: float
            - f1_macro: float
            - f1_weighted: float
            - confusion_matrix: np.ndarray  (shape 3×3, ordem LABELS)
            - report: str
            - y_true: List[str]
            - y_pred: List[str]
 
    Raises:
        ValueError: Se o classificador não for reconhecido ou não houver dados.
        FileNotFoundError: Se os artefatos do BERTimbau não forem encontrados.
    """
    if classificator not in AVAILABLE_CLASSIFICATORS:
        raise ValueError(
            f"Classificador '{classificator}' não reconhecido.\n"
            f"Opções disponíveis: {AVAILABLE_CLASSIFICATORS}"
        )

    y_true, y_pred = load_pairs(classificator, repo)
    f1 = compute_f1(y_true, y_pred)
 
    return {
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