"""Repositório para a tabela `dataset_split`.

Responsabilidade: persistir e consultar o particionamento estratificado do
dataset de tweets com anotação humana.

Estrutura do split:
    - split='test' + fold=NULL → hold-out fixo (~15%), usado na avaliação
                                   comparativa entre todos os modelos.
    - split='train' + fold=1..4 → pertence ao fold K do K-Fold estratificado, 
                                   usado exclusivamente no fine-tuning do BERTimbau.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from app.shared.db.database import DatabaseManager
from app.shared.db.tweets import TweetsRepository

_VALID_SENTIMENTS = ("positivo", "negativo", "neutro")
_N_SPLITS = 4


class DatasetSplitRepository(DatabaseManager):
    """Queries e operações sobre a tabela `dataset_split`."""

    # ── Escrita ───────────────────────────────────────────────────────────────

    def assign_split(
        self,
        n_splits: int = _N_SPLITS,
        test_size: float = 0.15,
        random_state: int = 42,
        force: bool = False,
    ) -> dict:
        """Persiste o particionamento estratificado no banco.

        Separa ~15% dos tweets anotados como hold-out (split='test') e
        distribui o restante em `n_splits` folds estratificados por sentimento
        (split='train', fold=1..n_splits).

        Idempotente: skipa se o split já foi atribuído. Use force=True para
        re-gerar (apaga as linhas existentes antes de reinserir).

        Args:
            n_splits: Número de folds do K-Fold (padrão 4).
            test_size: Proporção do hold-out (padrão 0.15).
            random_state: Semente para reprodutibilidade (padrão 42).
            force: Se True, apaga e recria o split existente.

        Returns:
            Dicionário com contagens:
            {'test': N, 'train': N, 'folds': {1: N, 2: N, ...}}.

        Raises:
            ValueError: Se não houver tweets humanos suficientes para o split.
        """
        from sklearn.model_selection import StratifiedKFold, train_test_split

        if not force and self.is_assigned():
            counts = self._count_splits()
            logger.info(
                f"Split já atribuído — use force=True para re-gerar. Contagens: {counts}"
            )
            return counts

        df = self._load_human_labeled_ids()

        if df.empty:
            raise ValueError(
                "Nenhum tweet com anotação humana válida encontrado. "
                "Execute `make annotate` antes de atribuir o split."
            )

        # 1. Hold-out estratificado
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["sentiment"],
            random_state=random_state,
        )

        # 2. K-Fold estratificado sobre o conjunto de treino
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        train_df = train_df.reset_index(drop=True)
        fold_series = pd.Series(0, index=train_df.index)
        for fold_num, (_, val_idx) in enumerate(
            skf.split(train_df, train_df["sentiment"]), start=1
        ):
            fold_series.iloc[val_idx] = fold_num
        train_df = train_df.copy()
        train_df["fold"] = fold_series

        # 3. Persiste
        rows: list[tuple] = []
        for _, row in test_df.iterrows():
            rows.append((int(row["tweet_id"]), "test", None))
        for _, row in train_df.iterrows():
            rows.append((int(row["tweet_id"]), "train", int(row["fold"])))

        if force:
            self._delete_all()
        self._bulk_insert(rows)

        fold_counts = {
            fold: int((train_df["fold"] == fold).sum())
            for fold in range(1, n_splits + 1)
        }
        counts = {
            "test": len(test_df),
            "train": len(train_df),
            "folds": fold_counts,
        }
        logger.info(f"Split atribuído com sucesso: {counts}")
        return counts

    # ── Leitura ───────────────────────────────────────────────────────────────

    def is_assigned(self) -> bool:
        """Retorna True se o split já foi persistido no banco."""
        query = "SELECT EXISTS(SELECT 1 FROM dataset_split LIMIT 1);"
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    result = cur.fetchone()
                    return bool(result[0]) if result else False
        except Exception as e:
            logger.error(f"Falha ao verificar split existente: {e}")
            return False

    def query_by_split(self, split: str) -> pd.DataFrame:
        """Retorna tweets do hold-out ('test') ou de todo o treino ('train').

        Args:
            split: 'test' ou 'train'.

        Returns:
            DataFrame com colunas: tweet_id, note_tweet, sentiment, fold.
            fold é NULL para todos os tweets do hold-out.
        """
        _validate_split(split)
        query = """
        SELECT
            t.id        AS tweet_id,
            t.note_tweet,
            tc.sentiment,
            ds.fold
        FROM dataset_split ds
        JOIN tweets t              ON t.id  = ds.tweet_id
        JOIN tweets_classification tc ON tc.tweet_id = t.id
            AND tc.classificator = 'Humano'
        WHERE ds.split = %s
        ORDER BY t.id;
        """
        return self._fetch_df(query, (split,), f"query_by_split('{split}')")

    def query_by_fold(self, fold: int) -> pd.DataFrame:
        """Retorna os tweets de um fold específico (conjunto de validação do K-Fold).

        Args:
            fold: Número do fold (1 a 4).

        Returns:
            DataFrame com colunas: tweet_id, note_tweet, sentiment, fold.
        """
        _validate_fold(fold)
        query = """
        SELECT
            t.id        AS tweet_id,
            t.note_tweet,
            tc.sentiment,
            ds.fold
        FROM dataset_split ds
        JOIN tweets t              ON t.id  = ds.tweet_id
        JOIN tweets_classification tc ON tc.tweet_id = t.id
            AND tc.classificator = 'Humano'
        WHERE ds.split = 'train'
          AND ds.fold  = %s
        ORDER BY t.id;
        """
        return self._fetch_df(query, (fold,), f"query_by_fold({fold})")

    def query_train_excluding_fold(self, fold: int) -> pd.DataFrame:
        """Retorna tweets de treino excluindo o fold indicado.

        Usado no K-Fold: para cada iteração, os folds restantes formam o
        conjunto de treinamento efetivo e o fold excluído é a validação.

        Args:
            fold: Número do fold a excluir (1 a 4).

        Returns:
            DataFrame com colunas: tweet_id, note_tweet, sentiment, fold.
        """
        _validate_fold(fold)
        query = """
        SELECT
            t.id        AS tweet_id,
            t.note_tweet,
            tc.sentiment,
            ds.fold
        FROM dataset_split ds
        JOIN tweets t              ON t.id  = ds.tweet_id
        JOIN tweets_classification tc ON tc.tweet_id = t.id
            AND tc.classificator = 'Humano'
        WHERE ds.split = 'train'
          AND ds.fold  != %s
        ORDER BY t.id;
        """
        return self._fetch_df(
            query, (fold,), f"query_train_excluding_fold({fold})"
        )

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _load_human_labeled_ids(self) -> pd.DataFrame:
        """Retorna (tweet_id, sentiment) para todos os tweets financeiros rotulados."""
        try:
            df = TweetsRepository().query_all_tweets_with_human_classification()
            df = df[df["is_finance_tweet"] == 1]
            df = df[df["sentiment"].isin(_VALID_SENTIMENTS)]
            return df[["id", "sentiment"]].rename(columns={"id": "tweet_id"})
        except Exception as e:
            logger.error(f"Falha ao carregar dados para split: {e}")
            return pd.DataFrame()

    def _bulk_insert(self, rows: list[tuple]) -> None:
        """Insere linhas (tweet_id, split, fold) em dataset_split."""
        query = """
        INSERT INTO dataset_split (tweet_id, split, fold)
        VALUES (%s, %s, %s)
        ON CONFLICT (tweet_id) DO NOTHING;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(query, rows)
                    conn.commit()
                    logger.debug(f"_bulk_insert: {len(rows)} linhas inseridas.")
        except Exception as e:
            logger.error(f"Falha no bulk insert de dataset_split: {e}")
            conn.rollback()

    def _delete_all(self) -> None:
        """Remove todas as linhas de dataset_split (usado com force=True)."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM dataset_split;")
                    conn.commit()
                    logger.debug("_delete_all: dataset_split limpo.")
        except Exception as e:
            logger.error(f"Falha ao limpar dataset_split: {e}")
            conn.rollback()

    def _count_splits(self) -> dict:
        """Retorna contagem de tweets por split e fold no banco."""
        query = """
        SELECT split, fold, COUNT(*)
        FROM dataset_split
        GROUP BY split, fold
        ORDER BY split, fold;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
            test_count = sum(c for s, f, c in rows if s == "test")
            train_count = sum(c for s, f, c in rows if s == "train")
            folds = {f: c for s, f, c in rows if s == "train" and f is not None}
            return {"test": test_count, "train": train_count, "folds": folds}
        except Exception as e:
            logger.error(f"Falha ao contar splits: {e}")
            return {}

    def _fetch_df(self, query: str, params: tuple, label: str) -> pd.DataFrame:
        """Executa uma query de leitura e retorna DataFrame padronizado."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    logger.info(f"{label}: {len(results)} tweets.")
                    return pd.DataFrame(
                        results,
                        columns=["tweet_id", "note_tweet", "sentiment", "fold"],
                    )
        except Exception as e:
            logger.error(f"Falha em {label}: {e}")
            return pd.DataFrame()


# ── Validadores ───────────────────────────────────────────────────────────────

def _validate_split(split: str) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"split inválido: '{split}'. Use 'train' ou 'test'.")


def _validate_fold(fold: int) -> None:
    if fold not in range(1, _N_SPLITS + 1):
        raise ValueError(
            f"fold inválido: {fold}. Use um valor entre 1 e {_N_SPLITS}."
        )
    

if __name__ == "__main__":
    repo = DatasetSplitRepository()
    counts = repo.assign_split(force=True)
    print(counts)