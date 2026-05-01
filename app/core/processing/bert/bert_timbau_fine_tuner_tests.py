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

from app.core.processing.bert.bert_timbau_fine_tuner import (
    preprocess,
    TweetDataset,
    compute_metrics,
    stratified_split,
    load_labeled_data,
    LABEL_TO_ID,
    ID_TO_LABEL,
    MAX_LENGTH,
    BASE_MODEL,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def labeled_df():
    """DataFrame mínimo com tweets rotulados para testes de split."""
    rows = []
    for i in range(90):
        sentiment = ["positivo", "negativo", "neutro"][i % 3]
        rows.append({
            "id": i + 1,
            "tweet_id": str(1000 + i),
            "note_tweet": f"Tweet de teste número {i}",
            "note_tweet_clean": f"Tweet teste {i}",
            "sentiment": sentiment,
            "label_id": LABEL_TO_ID[sentiment],
            "is_finance_tweet": 1,
            "has_human_classification": True,
        })
    return pd.DataFrame(rows)


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
        item = ds[0]
        assert "labels" in item

    def test_getitem_contains_input_ids(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [0, 1])
        item = ds[0]
        assert "input_ids" in item

    def test_getitem_contains_attention_mask(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [0, 1])
        item = ds[0]
        assert "attention_mask" in item

    def test_label_value_at_index(self, mock_encodings):
        ds = TweetDataset(mock_encodings, [2, 1])
        item = ds[0]
        # Converte para int para comparar independente do tipo tensor
        assert int(item["labels"]) == 2

    def test_empty_dataset_has_len_zero(self):
        ds = TweetDataset({"input_ids": [], "attention_mask": []}, [])
        assert len(ds) == 0


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_dict_with_expected_keys(self):
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        labels = np.array([0, 2])
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.accuracy_score", return_value=1.0), \
             patch("app.core.processing.bert.bert_timbau_fine_tuner.f1_score", return_value=1.0):
            result = compute_metrics((logits, labels))
        assert set(result.keys()) == {"accuracy", "f1_macro", "f1_weighted"}

    def test_returns_floats(self):
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        labels = np.array([0, 2])
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.accuracy_score", return_value=1.0), \
             patch("app.core.processing.bert.bert_timbau_fine_tuner.f1_score", return_value=1.0):
            result = compute_metrics((logits, labels))
        for v in result.values():
            assert isinstance(v, float)

    def test_argmax_selects_correct_class(self):
        # logits[0] → classe 0 (2.0 > 0.5 > 0.1)
        # logits[1] → classe 2 (2.0 > 0.5 > 0.1)
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
        labels = np.array([0, 2])
        captured_preds = {}

        def mock_accuracy(y_true, y_pred):
            captured_preds["preds"] = y_pred
            return 1.0

        with patch("app.core.processing.bert.bert_timbau_fine_tuner.accuracy_score", side_effect=mock_accuracy), \
             patch("app.core.processing.bert.bert_timbau_fine_tuner.f1_score", return_value=1.0):
            compute_metrics((logits, labels))

        np.testing.assert_array_equal(captured_preds["preds"], [0, 2])


# ── stratified_split ──────────────────────────────────────────────────────────

class TestStratifiedSplit:
    def test_returns_three_dataframes(self, labeled_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.train_test_split",
                   side_effect=[
                       (labeled_df.iloc[:63], labeled_df.iloc[63:]),
                       (labeled_df.iloc[63:76], labeled_df.iloc[76:]),
                   ]):
            result = stratified_split(labeled_df)
        assert len(result) == 3

    def test_split_sizes_sum_to_total(self, labeled_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.train_test_split",
                   side_effect=[
                       (labeled_df.iloc[:63].reset_index(drop=True),
                        labeled_df.iloc[63:].reset_index(drop=True)),
                       (labeled_df.iloc[63:76].reset_index(drop=True),
                        labeled_df.iloc[76:].reset_index(drop=True)),
                   ]):
            train, val, test = stratified_split(labeled_df)
        assert len(train) + len(val) + len(test) == len(labeled_df)

    def test_each_split_is_dataframe(self, labeled_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.train_test_split",
                   side_effect=[
                       (labeled_df.iloc[:63].reset_index(drop=True),
                        labeled_df.iloc[63:].reset_index(drop=True)),
                       (labeled_df.iloc[63:76].reset_index(drop=True),
                        labeled_df.iloc[76:].reset_index(drop=True)),
                   ]):
            train, val, test = stratified_split(labeled_df)
        for split in (train, val, test):
            assert isinstance(split, pd.DataFrame)

    def test_passes_stratify_param(self, labeled_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.train_test_split",
                   side_effect=[
                       (labeled_df.iloc[:63].reset_index(drop=True),
                        labeled_df.iloc[63:].reset_index(drop=True)),
                       (labeled_df.iloc[63:76].reset_index(drop=True),
                        labeled_df.iloc[76:].reset_index(drop=True)),
                   ]) as mock_split:
            stratified_split(labeled_df)
        first_call_kwargs = mock_split.call_args_list[0][1]
        assert "stratify" in first_call_kwargs


# ── load_labeled_data ─────────────────────────────────────────────────────────

class TestLoadLabeledData:
    @pytest.fixture
    def mock_repo_df(self):
        return pd.DataFrame([
            {"id": 1, "tweet_id": "a", "username": "59773459",
             "note_tweet": "PETR4 em alta", "created_at": "2025-10-01",
             "likes": 5, "hashtags": None, "tweet": None,
             "sentiment": "positivo", "is_finance_tweet": 1,
             "has_human_classification": True},
            {"id": 2, "tweet_id": "b", "username": "59773459",
             "note_tweet": "IBOV cai 2%", "created_at": "2025-10-02",
             "likes": 3, "hashtags": None, "tweet": None,
             "sentiment": "negativo", "is_finance_tweet": 1,
             "has_human_classification": True},
            {"id": 3, "tweet_id": "c", "username": "59773459",
             "note_tweet": "Volume estável", "created_at": "2025-10-03",
             "likes": 1, "hashtags": None, "tweet": None,
             "sentiment": "invalido", "is_finance_tweet": 1,
             "has_human_classification": True},
        ])

    def test_returns_dataframe(self, mock_repo_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = mock_repo_df
            result = load_labeled_data()
        assert isinstance(result, pd.DataFrame)

    def test_filters_invalid_sentiments(self, mock_repo_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = mock_repo_df
            result = load_labeled_data()
        assert "invalido" not in result["sentiment"].values

    def test_contains_only_valid_labels(self, mock_repo_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = mock_repo_df
            result = load_labeled_data()
        assert set(result["sentiment"].unique()).issubset(set(LABEL_TO_ID.keys()))

    def test_adds_note_tweet_clean_column(self, mock_repo_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = mock_repo_df
            result = load_labeled_data()
        assert "note_tweet_clean" in result.columns

    def test_adds_label_id_column(self, mock_repo_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = mock_repo_df
            result = load_labeled_data()
        assert "label_id" in result.columns

    def test_label_id_matches_label_to_id(self, mock_repo_df):
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = mock_repo_df
            result = load_labeled_data()
        for _, row in result.iterrows():
            assert row["label_id"] == LABEL_TO_ID[row["sentiment"]]

    def test_drops_empty_cleaned_tweets(self):
        df = pd.DataFrame([{
            "id": 1, "tweet_id": "a", "username": "x",
            "note_tweet": "   ", "created_at": "2025-10-01",
            "likes": 0, "hashtags": None, "tweet": None,
            "sentiment": "positivo", "is_finance_tweet": 1,
            "has_human_classification": True,
        }])
        with patch("app.core.processing.bert.bert_timbau_fine_tuner.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = df
            result = load_labeled_data()
        assert result.empty