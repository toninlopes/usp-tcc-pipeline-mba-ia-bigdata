from unittest.mock import MagicMock, patch
from typing import List
import numpy as np
import pandas as pd
import pytest

from app.core.evaluation.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_f1,
    compute_report,
    evaluate,
    load_pairs,
    print_evaluation,
    LABELS,
)


# ── Fixtures compartilhadas ───────────────────────────────────────────────────

@pytest.fixture
def perfect_labels():
    """Predições perfeitas — modelo concorda 100% com humano."""
    y_true = ["positivo", "neutro", "negativo", "positivo", "negativo"]
    y_pred = ["positivo", "neutro", "negativo", "positivo", "negativo"]
    return y_true, y_pred


@pytest.fixture
def imperfect_labels():
    """Predições com erros mistos."""
    y_true = ["positivo", "neutro", "negativo", "positivo", "negativo"]
    y_pred = ["positivo", "negativo", "neutro", "positivo", "negativo"]
    return y_true, y_pred


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.query_classification_pairs.return_value = pd.DataFrame({
        "tweet_id": [1, 2, 3, 4, 5],
        "human_label": ["positivo", "neutro", "negativo", "positivo", "negativo"],
        "model_label": ["positivo", "neutro", "negativo", "positivo", "negativo"],
    })
    return repo


# ── compute_accuracy ──────────────────────────────────────────────────────────

class TestComputeAccuracy:
    def test_perfect_prediction(self, perfect_labels):
        y_true, y_pred = perfect_labels
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        y_true = ["positivo", "positivo"]
        y_pred = ["negativo", "negativo"]
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_partial_accuracy(self, imperfect_labels):
        y_true, y_pred = imperfect_labels
        result = compute_accuracy(y_true, y_pred)
        assert 0.0 < result < 1.0

    def test_returns_float(self, perfect_labels):
        y_true, y_pred = perfect_labels
        assert isinstance(compute_accuracy(y_true, y_pred), float)

    def test_single_sample(self):
        assert compute_accuracy(["positivo"], ["positivo"]) == 1.0
        assert compute_accuracy(["positivo"], ["negativo"]) == 0.0


# ── compute_f1 ────────────────────────────────────────────────────────────────

class TestComputeF1:
    def test_perfect_prediction_returns_one(self, perfect_labels):
        y_true, y_pred = perfect_labels
        f1 = compute_f1(y_true, y_pred)
        assert f1["macro"] == pytest.approx(1.0)
        assert f1["weighted"] == pytest.approx(1.0)

    def test_returns_macro_and_weighted_keys(self, perfect_labels):
        y_true, y_pred = perfect_labels
        f1 = compute_f1(y_true, y_pred)
        assert set(f1.keys()) == {"macro", "weighted"}

    def test_values_between_zero_and_one(self, imperfect_labels):
        y_true, y_pred = imperfect_labels
        f1 = compute_f1(y_true, y_pred)
        assert 0.0 <= f1["macro"] <= 1.0
        assert 0.0 <= f1["weighted"] <= 1.0

    def test_returns_floats(self, perfect_labels):
        y_true, y_pred = perfect_labels
        f1 = compute_f1(y_true, y_pred)
        assert isinstance(f1["macro"], float)
        assert isinstance(f1["weighted"], float)

    def test_missing_class_does_not_raise(self):
        # Apenas "positivo" presente — zero_division=0 evita erro
        y_true = ["positivo", "positivo"]
        y_pred = ["positivo", "positivo"]
        f1 = compute_f1(y_true, y_pred)
        assert f1["macro"] >= 0.0


# ── compute_confusion_matrix ──────────────────────────────────────────────────

class TestComputeConfusionMatrix:
    def test_returns_numpy_array(self, perfect_labels):
        y_true, y_pred = perfect_labels
        cm = compute_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, np.ndarray)

    def test_shape_is_3x3(self, perfect_labels):
        y_true, y_pred = perfect_labels
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_diagonal_is_correct_count_on_perfect_pred(self):
        y_true = ["positivo", "neutro", "negativo"]
        y_pred = ["positivo", "neutro", "negativo"]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm[0, 0] == 1  # positivo
        assert cm[1, 1] == 1  # neutro
        assert cm[2, 2] == 1  # negativo

    def test_off_diagonal_captures_errors(self):
        # positivo predito como negativo
        y_true = ["positivo"]
        y_pred = ["negativo"]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm[0, 0] == 0   # positivo correto
        assert cm[0, 2] == 1   # positivo → negativo

    def test_label_order_matches_labels_constant(self):
        # LABELS = ["positivo", "neutro", "negativo"]
        # linha 0 = positivo, coluna 2 = negativo
        y_true = ["positivo"]
        y_pred = ["negativo"]
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm[LABELS.index("positivo"), LABELS.index("negativo")] == 1


# ── compute_report ────────────────────────────────────────────────────────────

class TestComputeReport:
    def test_returns_string(self, perfect_labels):
        y_true, y_pred = perfect_labels
        assert isinstance(compute_report(y_true, y_pred), str)

    def test_contains_all_labels(self, perfect_labels):
        y_true, y_pred = perfect_labels
        report = compute_report(y_true, y_pred)
        for label in LABELS:
            assert label in report

    def test_contains_precision_recall_f1(self, perfect_labels):
        y_true, y_pred = perfect_labels
        report = compute_report(y_true, y_pred)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report


# ── load_pairs ────────────────────────────────────────────────────────────────

class TestLoadPairs:
    def test_returns_tuple_of_two_lists(self, mock_repo):
        y_true, y_pred = load_pairs("FinBERT-PT-BR", repo=mock_repo)
        assert isinstance(y_true, list)
        assert isinstance(y_pred, list)

    def test_returns_correct_labels(self, mock_repo):
        y_true, y_pred = load_pairs("FinBERT-PT-BR", repo=mock_repo)
        assert y_true == ["positivo", "neutro", "negativo", "positivo", "negativo"]
        assert y_pred == ["positivo", "neutro", "negativo", "positivo", "negativo"]

    def test_same_length(self, mock_repo):
        y_true, y_pred = load_pairs("FinBERT-PT-BR", repo=mock_repo)
        assert len(y_true) == len(y_pred)

    def test_raises_when_no_pairs(self):
        repo = MagicMock()
        repo.query_classification_pairs.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="Nenhum par encontrado"):
            load_pairs("FinBERT-PT-BR", repo=repo)

    def test_passes_classificator_to_repo(self, mock_repo):
        load_pairs("SentiLex-PT", repo=mock_repo)
        mock_repo.query_classification_pairs.assert_called_once_with("SentiLex-PT")


# ── evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_dict(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert isinstance(result, dict)

    def test_contains_all_keys(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        expected_keys = {
            "classificator", "n_samples", "accuracy",
            "f1_macro", "f1_weighted", "confusion_matrix",
            "report", "y_true", "y_pred",
        }
        assert set(result.keys()) == expected_keys

    def test_classificator_matches_input(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert result["classificator"] == "FinBERT-PT-BR"

    def test_n_samples_is_correct(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert result["n_samples"] == 5

    def test_perfect_accuracy_on_perfect_pairs(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_f1_macro_between_zero_and_one(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert 0.0 <= result["f1_macro"] <= 1.0

    def test_confusion_matrix_shape(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert result["confusion_matrix"].shape == (3, 3)

    def test_report_is_string(self, mock_repo):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        assert isinstance(result["report"], str)

    def test_raises_when_no_pairs(self):
        repo = MagicMock()
        repo.query_classification_pairs.return_value = pd.DataFrame()
        with pytest.raises(ValueError):
            evaluate("FinBERT-PT-BR", repo=repo)


# ── print_evaluation ──────────────────────────────────────────────────────────

class TestPrintEvaluation:
    def test_does_not_raise(self, mock_repo, capsys):
        result = evaluate("FinBERT-PT-BR", repo=mock_repo)
        print_evaluation(result)
        captured = capsys.readouterr()
        assert "FinBERT-PT-BR" in captured.out
        assert "Acurácia" in captured.out
        assert "F1 Macro" in captured.out