from unittest.mock import MagicMock, patch
from typing import Dict
import numpy as np
import pandas as pd
import pytest
import sys

# Bloqueia dependências pesadas antes de qualquer import do módulo
for _mod in [
    "torch",
    "torch.utils",
    "torch.utils.data",
    "torch.nn",
    "sklearn",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.utils",
    "sklearn.utils.class_weight",
    "transformers",
    "transformers.trainer",
    "transformers.trainer_callback",
    "transformers.training_args",
    "transformers.models",
    "transformers.models.auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
]:
    sys.modules.setdefault(_mod, MagicMock())

from app.core.classification.bert.bert_timbau_fine_tuner import (
    preprocess,
    prepare_df,
    TweetDataset,
    compute_metrics,
    LABEL_TO_ID,
    ID_TO_LABEL,
    MAX_LENGTH,
    BASE_MODEL,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame no formato retornado pelo DatasetSplitRepository."""
    return pd.DataFrame([
        {"tweet_id": 1, "note_tweet": "PETR4 em alta https://t.co/abc", "sentiment": "positivo", "fold": None},
        {"tweet_id": 2, "note_tweet": "IBOV cai 2%",                    "sentiment": "negativo", "fold": None},
        {"tweet_id": 3, "note_tweet": "Volume estável no mercado",       "sentiment": "neutro",   "fold": None},
        {"tweet_id": 4, "note_tweet": "Tweet sem sentimento válido",     "sentiment": "invalido", "fold": None},
    ])


@pytest.fixture
def mock_encodings():
    """Encodings simulados no formato retornado pelo tokenizer."""
    return {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
    }


# ── Constantes ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_label_to_id_has_three_classes(self):
        assert len(LABEL_TO_ID) == 3

    def test_label_to_id_contains_all_sentiments(self):
        assert set(LABEL_TO_ID.keys()) == {"positivo", "negativo", "neutro"}

    def test_id_to_label_is_inverse_of_label_to_id(self):
        for label, id_ in LABEL_TO_ID.items():
            assert ID_TO_LABEL[id_] == label

    def test_max_length_is_128(self):
        assert MAX_LENGTH == 128

    def test_base_model_is_bertimbau(self):
        assert "bert-base-portuguese-cased" in BASE_MODEL


# ── preprocess ────────────────────────────────────────────────────────────────

class TestPreprocess:
    def test_removes_url(self):
        assert "https://" not in preprocess("Veja https://t.co/abc")

    def test_replaces_mention(self):
        assert "@InfoMoney" not in preprocess("Via @InfoMoney")

    def test_removes_hashtag_symbol(self):
        result = preprocess("Alta do #IBOV")
        assert "#" not in result
        assert "IBOV" in result

    def test_does_not_lowercase(self):
        """BERTimbau é cased — capitalização deve ser preservada no fine-tuning."""
        result = preprocess("PETR4 em Alta")
        assert "PETR4" in result
        assert "Alta" in result

    def test_collapses_spaces(self):
        assert "  " not in preprocess("texto   espaçado")

    def test_empty_string_returns_empty(self):
        assert preprocess("") == ""

    def test_preserves_accents(self):
        result = preprocess("Inflação e taxa Selic")
        assert "Inflação" in result
        assert "Selic" in result


# ── prepare_df ────────────────────────────────────────────────────────────────

class TestPrepareDF:
    def test_returns_dataframe(self, sample_df):
        assert isinstance(prepare_df(sample_df), pd.DataFrame)

    def test_filters_invalid_sentiments(self, sample_df):
        result = prepare_df(sample_df)
        assert "invalido" not in result["sentiment"].values

    def test_contains_only_valid_labels(self, sample_df):
        result = prepare_df(sample_df)
        assert set(result["sentiment"].unique()).issubset(set(LABEL_TO_ID.keys()))

    def test_adds_note_tweet_clean_column(self, sample_df):
        assert "note_tweet_clean" in prepare_df(sample_df).columns

    def test_adds_label_id_column(self, sample_df):
        assert "label_id" in prepare_df(sample_df).columns

    def test_label_id_matches_label_to_id(self, sample_df):
        result = prepare_df(sample_df)
        for _, row in result.iterrows():
            assert row["label_id"] == LABEL_TO_ID[row["sentiment"]]

    def test_drops_empty_cleaned_tweets(self):
        df = pd.DataFrame([
            {"tweet_id": 1, "note_tweet": "   ", "sentiment": "positivo", "fold": None},
        ])
        assert prepare_df(df).empty

    def test_does_not_mutate_input(self, sample_df):
        original_len = len(sample_df)
        prepare_df(sample_df)
        assert len(sample_df) == original_len

    def test_index_is_reset(self, sample_df):
        result = prepare_df(sample_df)
        assert list(result.index) == list(range(len(result)))


# ── TweetDataset ──────────────────────────────────────────────────────────────

class TestTweetDataset:
    def test_len_matches_labels(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [0, 1])
        assert len(ds) == 2

    def test_getitem_returns_dict(self, mock_encodings):
        import torch
        ds = TweetDataset(mock_encodings, [0, 1])
        item = ds[0]
        assert isinstance(item, dict)

    def test_getitem_contains_labels_key(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [0, 1])
        assert "labels" in ds[0]

    def test_getitem_contains_input_ids(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [0, 1])
        assert "input_ids" in ds[0]

    def test_getitem_contains_attention_mask(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [0, 1])
        assert "attention_mask" in ds[0]

    def test_label_value_at_index(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [2, 1])
        assert int(ds[0]["labels"]) == 2

    def test_empty_dataset_has_len_zero(self):
        ds = TweetDataset({"input_ids": [], "attention_mask": []}, [])
        assert len(ds) == 0


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_dict_with_expected_keys(self):
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        labels = np.array([0, 2])
        with patch("app.core.classification.bert.bert_timbau_fine_tuner.accuracy_score", return_value=1.0), \
             patch("app.core.classification.bert.bert_timbau_fine_tuner.f1_score", return_value=1.0):
            result = compute_metrics((logits, labels))
        assert set(result.keys()) == {"accuracy", "f1_macro", "f1_weighted"}

    def test_returns_floats(self):
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        labels = np.array([0, 2])
        with patch("app.core.classification.bert.bert_timbau_fine_tuner.accuracy_score", return_value=1.0), \
             patch("app.core.classification.bert.bert_timbau_fine_tuner.f1_score", return_value=1.0):
            result = compute_metrics((logits, labels))
        for v in result.values():
            assert isinstance(v, float)

    def test_argmax_selects_correct_class(self):
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        labels = np.array([0, 2])
        captured_preds = {}

        def mock_accuracy(y_true, y_pred):
            captured_preds["preds"] = y_pred
            return 1.0

        with patch("app.core.classification.bert.bert_timbau_fine_tuner.accuracy_score", side_effect=mock_accuracy), \
             patch("app.core.classification.bert.bert_timbau_fine_tuner.f1_score", return_value=1.0):
            compute_metrics((logits, labels))

        np.testing.assert_array_equal(captured_preds["preds"], [0, 2])
