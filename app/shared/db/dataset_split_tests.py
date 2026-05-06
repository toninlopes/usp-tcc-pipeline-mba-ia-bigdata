from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from app.shared.db.dataset_split import DatasetSplitRepository, _N_SPLITS, _VALID_SENTIMENTS
from app.shared.test_helpers import make_repo, mock_get_connection


# ── Fixtures de módulo ────────────────────────────────────────────────────────

@pytest.fixture
def repo():
    return make_repo(DatasetSplitRepository)


@pytest.fixture
def labeled_rows():
    """90 tweets rotulados balanceados: 30 por classe."""
    rows = []
    for i in range(90):
        sentiment = ["positivo", "negativo", "neutro"][i % 3]
        rows.append((i + 1, sentiment))
    return rows


@pytest.fixture
def labeled_df(labeled_rows):
    return pd.DataFrame(labeled_rows, columns=["tweet_id", "sentiment"])


@pytest.fixture
def split_row_test():
    return (1, "Mercado em alta.", "positivo", None)


@pytest.fixture
def split_row_train():
    return (2, "Selic mantida.", "neutro", 1)


# ── is_assigned ───────────────────────────────────────────────────────────────

class TestIsAssigned:
    def test_returns_false_when_table_empty(self, repo, cursor):
        cursor.fetchone.return_value = (False,)
        mock_get_connection(repo, cursor)
        assert repo.is_assigned() is False

    def test_returns_true_when_rows_exist(self, repo, cursor):
        cursor.fetchone.return_value = (True,)
        mock_get_connection(repo, cursor)
        assert repo.is_assigned() is True

    def test_returns_false_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        assert repo.is_assigned() is False


# ── _load_human_labeled_ids ───────────────────────────────────────────────────

class TestLoadHumanLabeledIds:
    @pytest.fixture
    def full_df(self):
        """DataFrame simulando query_all_tweets_with_human_classification."""
        return pd.DataFrame([
            {"id": 1, "sentiment": "positivo", "is_finance_tweet": 1, "has_human_classification": True},
            {"id": 2, "sentiment": "negativo", "is_finance_tweet": 1, "has_human_classification": True},
            {"id": 3, "sentiment": None,       "is_finance_tweet": 0, "has_human_classification": False},
            {"id": 4, "sentiment": "neutro",   "is_finance_tweet": 1, "has_human_classification": True},
            {"id": 5, "sentiment": "positivo", "is_finance_tweet": 0, "has_human_classification": False},
        ])

    def test_returns_only_finance_tweets(self, repo, full_df):
        with patch("app.shared.db.dataset_split.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = full_df
            result = repo._load_human_labeled_ids()
        assert set(result["tweet_id"].tolist()) == {1, 2, 4}

    def test_excludes_non_finance_tweets(self, repo, full_df):
        with patch("app.shared.db.dataset_split.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = full_df
            result = repo._load_human_labeled_ids()
        assert 3 not in result["tweet_id"].values
        assert 5 not in result["tweet_id"].values

    def test_returns_correct_columns(self, repo, full_df):
        with patch("app.shared.db.dataset_split.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = full_df
            result = repo._load_human_labeled_ids()
        assert list(result.columns) == ["tweet_id", "sentiment"]

    def test_returns_only_valid_sentiments(self, repo, full_df):
        with patch("app.shared.db.dataset_split.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.return_value = full_df
            result = repo._load_human_labeled_ids()
        assert result["sentiment"].isin(_VALID_SENTIMENTS).all()

    def test_returns_empty_on_exception(self, repo):
        with patch("app.shared.db.dataset_split.TweetsRepository") as MockRepo:
            MockRepo.return_value.query_all_tweets_with_human_classification.side_effect = Exception("db error")
            result = repo._load_human_labeled_ids()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ── assign_split ──────────────────────────────────────────────────────────────

class TestAssignSplit:
    """Testa assign_split mockando os métodos internos diretamente.

    assign_split abre múltiplas conexões em sequência (is_assigned,
    _load_human_labeled_ids, _bulk_insert). Mockar o cursor via
    mock_get_connection causaria estado inconsistente entre as chamadas.
    A abordagem correta é mockar cada método interno individualmente.
    """

    @pytest.fixture
    def repo_unassigned(self, repo, labeled_df):
        repo.is_assigned = MagicMock(return_value=False)
        repo._load_human_labeled_ids = MagicMock(return_value=labeled_df)
        repo._bulk_insert = MagicMock()
        repo._delete_all = MagicMock()
        return repo

    def test_returns_dict_with_expected_keys(self, repo_unassigned):
        result = repo_unassigned.assign_split()
        assert set(result.keys()) == {"test", "train", "folds"}

    def test_folds_dict_has_n_splits_keys(self, repo_unassigned):
        result = repo_unassigned.assign_split()
        assert set(result["folds"].keys()) == set(range(1, _N_SPLITS + 1))

    def test_total_sums_to_input(self, repo_unassigned, labeled_rows):
        result = repo_unassigned.assign_split()
        assert result["test"] + result["train"] == len(labeled_rows)

    def test_fold_counts_sum_to_train(self, repo_unassigned):
        result = repo_unassigned.assign_split()
        assert sum(result["folds"].values()) == result["train"]

    def test_test_is_approximately_15_percent(self, repo_unassigned, labeled_rows):
        result = repo_unassigned.assign_split()
        ratio = result["test"] / len(labeled_rows)
        assert 0.10 <= ratio <= 0.20

    def test_train_is_largest_partition(self, repo_unassigned):
        result = repo_unassigned.assign_split()
        assert result["train"] > result["test"]

    def test_skips_when_already_assigned(self, repo):
        expected = {"test": 13, "train": 77, "folds": {1: 19, 2: 19, 3: 20, 4: 19}}
        repo.is_assigned = MagicMock(return_value=True)
        repo._count_splits = MagicMock(return_value=expected)
        result = repo.assign_split(force=False)
        assert result == expected
        repo._count_splits.assert_called_once()

    def test_skips_does_not_call_load_or_insert(self, repo):
        repo.is_assigned = MagicMock(return_value=True)
        repo._count_splits = MagicMock(return_value={})
        repo._load_human_labeled_ids = MagicMock()
        repo._bulk_insert = MagicMock()
        repo.assign_split(force=False)
        repo._load_human_labeled_ids.assert_not_called()
        repo._bulk_insert.assert_not_called()

    def test_force_ignores_existing_split(self, repo, labeled_df, labeled_rows):
        repo.is_assigned = MagicMock(return_value=True)
        repo._load_human_labeled_ids = MagicMock(return_value=labeled_df)
        repo._bulk_insert = MagicMock()
        repo._delete_all = MagicMock()
        result = repo.assign_split(force=True)
        repo._delete_all.assert_called_once()
        assert result["test"] + result["train"] == len(labeled_rows)

    def test_raises_when_no_labeled_tweets(self, repo):
        repo.is_assigned = MagicMock(return_value=False)
        repo._load_human_labeled_ids = MagicMock(return_value=pd.DataFrame())
        with pytest.raises(ValueError, match="Nenhum tweet"):
            repo.assign_split()

    def test_deterministic_with_same_seed(self, repo, labeled_df):
        repo.is_assigned = MagicMock(return_value=False)
        repo._load_human_labeled_ids = MagicMock(return_value=labeled_df.copy())
        repo._bulk_insert = MagicMock()
        r1 = repo.assign_split(random_state=42)

        repo.is_assigned = MagicMock(return_value=False)
        repo._load_human_labeled_ids = MagicMock(return_value=labeled_df.copy())
        repo._bulk_insert = MagicMock()
        r2 = repo.assign_split(random_state=42)

        assert r1["test"] == r2["test"]
        assert r1["train"] == r2["train"]
        assert r1["folds"] == r2["folds"]

    def test_custom_n_splits(self, repo, labeled_df):
        repo.is_assigned = MagicMock(return_value=False)
        repo._load_human_labeled_ids = MagicMock(return_value=labeled_df)
        repo._bulk_insert = MagicMock()
        result = repo.assign_split(n_splits=3)
        assert set(result["folds"].keys()) == {1, 2, 3}

    def test_bulk_insert_called_with_correct_splits(self, repo_unassigned, labeled_rows):
        repo_unassigned.assign_split()
        rows = repo_unassigned._bulk_insert.call_args[0][0]
        splits = {r[1] for r in rows}
        assert splits == {"train", "test"}

    def test_test_rows_have_null_fold(self, repo_unassigned):
        repo_unassigned.assign_split()
        rows = repo_unassigned._bulk_insert.call_args[0][0]
        test_rows = [r for r in rows if r[1] == "test"]
        assert all(r[2] is None for r in test_rows)

    def test_train_rows_have_valid_fold(self, repo_unassigned):
        repo_unassigned.assign_split()
        rows = repo_unassigned._bulk_insert.call_args[0][0]
        train_rows = [r for r in rows if r[1] == "train"]
        assert all(r[2] in range(1, _N_SPLITS + 1) for r in train_rows)


# ── query_by_split ────────────────────────────────────────────────────────────

class TestQueryBySplit:
    def test_returns_dataframe(self, repo, cursor, split_row_test):
        cursor.fetchall.return_value = [split_row_test]
        mock_get_connection(repo, cursor)
        result = repo.query_by_split("test")
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, split_row_test):
        cursor.fetchall.return_value = [split_row_test]
        mock_get_connection(repo, cursor)
        result = repo.query_by_split("test")
        assert list(result.columns) == ["tweet_id", "note_tweet", "sentiment", "fold"]

    def test_fold_is_none_for_test(self, repo, cursor, split_row_test):
        cursor.fetchall.return_value = [split_row_test]
        mock_get_connection(repo, cursor)
        result = repo.query_by_split("test")
        assert result.iloc[0]["fold"] is None

    def test_fold_is_set_for_train(self, repo, cursor, split_row_train):
        cursor.fetchall.return_value = [split_row_train]
        mock_get_connection(repo, cursor)
        result = repo.query_by_split("train")
        assert result.iloc[0]["fold"] == 1

    def test_passes_split_param_to_query(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_by_split("train")
        args = cursor.execute.call_args[0][1]
        assert args == ("train",)

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_by_split("test")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_accepts_valid_splits(self, repo, cursor, split):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        result = repo.query_by_split(split)
        assert isinstance(result, pd.DataFrame)

    def test_raises_on_invalid_split(self, repo):
        with pytest.raises(ValueError, match="split inválido"):
            repo.query_by_split("validacao")


# ── query_by_fold ─────────────────────────────────────────────────────────────

class TestQueryByFold:
    def test_returns_dataframe(self, repo, cursor, split_row_train):
        cursor.fetchall.return_value = [split_row_train]
        mock_get_connection(repo, cursor)
        result = repo.query_by_fold(1)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, split_row_train):
        cursor.fetchall.return_value = [split_row_train]
        mock_get_connection(repo, cursor)
        result = repo.query_by_fold(1)
        assert list(result.columns) == ["tweet_id", "note_tweet", "sentiment", "fold"]

    def test_passes_fold_param_to_query(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_by_fold(2)
        args = cursor.execute.call_args[0][1]
        assert args == (2,)

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_by_fold(1)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.parametrize("fold", [1, 2, 3, 4])
    def test_accepts_all_valid_folds(self, repo, cursor, fold):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        result = repo.query_by_fold(fold)
        assert isinstance(result, pd.DataFrame)

    def test_raises_on_invalid_fold(self, repo):
        with pytest.raises(ValueError, match="fold inválido"):
            repo.query_by_fold(5)

    def test_raises_on_zero_fold(self, repo):
        with pytest.raises(ValueError, match="fold inválido"):
            repo.query_by_fold(0)


# ── query_train_excluding_fold ────────────────────────────────────────────────

class TestQueryTrainExcludingFold:
    def test_returns_dataframe(self, repo, cursor, split_row_train):
        cursor.fetchall.return_value = [split_row_train]
        mock_get_connection(repo, cursor)
        result = repo.query_train_excluding_fold(1)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self, repo, cursor, split_row_train):
        cursor.fetchall.return_value = [split_row_train]
        mock_get_connection(repo, cursor)
        result = repo.query_train_excluding_fold(1)
        assert list(result.columns) == ["tweet_id", "note_tweet", "sentiment", "fold"]

    def test_passes_fold_param_to_query(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_train_excluding_fold(3)
        args = cursor.execute.call_args[0][1]
        assert args == (3,)

    def test_query_uses_not_equal_operator(self, repo, cursor):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        repo.query_train_excluding_fold(2)
        query = cursor.execute.call_args[0][0]
        assert "!=" in query

    def test_returns_empty_dataframe_on_exception(self, repo, cursor):
        cursor.execute.side_effect = Exception("db error")
        mock_get_connection(repo, cursor)
        result = repo.query_train_excluding_fold(1)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_raises_on_invalid_fold(self, repo):
        with pytest.raises(ValueError, match="fold inválido"):
            repo.query_train_excluding_fold(5)

    @pytest.mark.parametrize("fold", [1, 2, 3, 4])
    def test_accepts_all_valid_folds(self, repo, cursor, fold):
        cursor.fetchall.return_value = []
        mock_get_connection(repo, cursor)
        result = repo.query_train_excluding_fold(fold)
        assert isinstance(result, pd.DataFrame)