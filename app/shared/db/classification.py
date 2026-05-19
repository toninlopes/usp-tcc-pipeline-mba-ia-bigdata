import pandas as pd
from psycopg2.extras import Json
from loguru import logger
from typing import Optional, List, Tuple
from collections import Counter, defaultdict

from app.shared.db.database import DatabaseManager


class ClassificationRepository(DatabaseManager):
    """Queries e operações sobre a tabela `tweets_classification`."""

    def insert_tweets_classification(
        self,
        tweet_id: int,
        is_finance_news: int,
        why_is_finance_news: str,
        sentiment: str,
        why_sentiment: str,
        classificator: str,
        score: Optional[float] = None,
    ) -> bool:
        """Insere uma classificação na tabela `tweets_classification`.

        Args:
            tweet_id: FK para tweets.id.
            is_finance_news: 1 para financeiro, 0 para não-financeiro.
            why_is_finance_news: Justificativa da classificação financeira.
            sentiment: Rótulo de sentimento ('positivo', 'negativo', 'neutro').
            why_sentiment: Justificativa do sentimento.
            classificator: Origem da classificação ('Humano' ou 'FinBERT-PT-BR').
            score: Score de confiança do modelo (apenas para classificações automáticas).

        Returns:
            True se inserido com sucesso, False caso contrário.
        """
        if hasattr(tweet_id, "item"):
            tweet_id = int(tweet_id.item())
        elif isinstance(tweet_id, (list, tuple)):
            tweet_id = int(tweet_id[0])
        else:
            tweet_id = int(tweet_id)

        query = """
        INSERT INTO tweets_classification (
            tweet_id,
            is_finance_news,
            why_is_finance_news,
            sentiment,
            why_sentiment,
            classificator,
            score
        ) VALUES (%s, %s, %s, lower(%s), %s, %s, %s)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            tweet_id,
                            int(is_finance_news),
                            why_is_finance_news,
                            sentiment,
                            why_sentiment,
                            classificator,
                            score,
                        ),
                    )
                    conn.commit()
                    logger.info(f"Classificação do tweet ID {tweet_id} inserida com sucesso.")
                    return True
        except Exception as e:
            logger.error(f"Falha ao inserir classificação do tweet ID {tweet_id}: {e}")
            conn.rollback()
            return False

    def query_tweets_classification_by_id(self, tweet_id: int) -> pd.DataFrame:
        """Retorna todas as classificações de um tweet específico.

        Args:
            tweet_id: FK para tweets.id.
        """
        if hasattr(tweet_id, "item"):
            tweet_id = int(tweet_id.item())
        elif isinstance(tweet_id, (list, tuple)):
            tweet_id = int(tweet_id[0])
        else:
            tweet_id = int(tweet_id)

        query = """
        SELECT * FROM tweets_classification
        WHERE tweet_id = %s
        ORDER BY id DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (tweet_id,))
                    results = cur.fetchall()
                    logger.info(f"Classificações do tweet ID {tweet_id} consultadas com sucesso.")
                    return pd.DataFrame(
                        results,
                        columns=[
                            "id",
                            "tweet_id",
                            "sentiment",
                            "why_sentiment",
                            "is_finance_news",
                            "why_is_finance_news",
                            "classificator",
                            "score",
                        ],
                    )
        except Exception as e:
            logger.error(f"Falha ao consultar classificações do tweet ID {tweet_id}: {e}")
            return pd.DataFrame()
        
    def query_classification_pairs(
        self,
        model_classificator: str,
        split: Optional[str] = "teste",
    ) -> pd.DataFrame:
        """Retorna pares de classificação (Humano, Modelo) para o mesmo tweet.

        Usado pela avaliação para comparar o gold standard humano com
        as predições de um modelo específico.

        Args:
            model_classificator: Nome do classificador a comparar com 'Humano'.
                                Ex: 'FinBERT-PT-BR', 'SentiLex-PT', 'OpLexicon'.
            split: Partição do dataset a usar — 'treino', 'validacao' ou 'teste'.
                   Padrão 'teste' (hold-out). Passe None para usar todos os tweets
                   anotados sem filtro de split (útil no dashboard).

        Returns:
            DataFrame com colunas tweet_id, human_label, model_label.
        """
        split_clause = "AND h.split = %s" if split else ""
        query = f"""
        SELECT
            h.tweet_id,
            h.sentiment AS human_label,
            m.sentiment AS model_label
        FROM tweets_classification h
        JOIN tweets_classification m ON h.tweet_id = m.tweet_id
        WHERE h.classificator = 'Humano'
          AND m.classificator = %s
          {split_clause}
        ORDER BY h.tweet_id;
        """
        params = (model_classificator, split) if split else (model_classificator,)
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    logger.info(
                        f"Pares Humano vs {model_classificator}"
                        f"{f' (split={split})' if split else ''}: {len(results)} encontrados."
                    )
                    return pd.DataFrame(
                        results,
                        columns=["tweet_id", "human_label", "model_label"],
                    )
        except Exception as e:
            logger.error(f"Falha ao buscar pares de classificação: {e}")
            return pd.DataFrame()

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def bootstrap_human_labels(self, force: bool = False) -> dict:
        """Insere classificações 'Humano' derivadas das classificações automáticas.

        Para cada tweet que possui pelo menos uma classificação automática:
        - Se a maioria dos classificadores marcou is_finance_news = 0: insere
          um registro Humano com is_finance_news = 0 e sentiment = NULL.
        - Se a maioria marcou is_finance_news = 1: insere um registro Humano
          com is_finance_news = 1 e o sentimento mais frequente entre os
          classificadores automáticos.

        Idempotente: ignora tweets que já possuem classificação Humano.
        Use force=True para apagar os registros Humano existentes e recriar.

        Args:
            force: Se True, apaga os registros Humano existentes antes de recriar.

        Returns:
            Dicionário {'inserted': N, 'skipped': N}.
        """
        if force:
            self._delete_human_labels()

        model_rows = self._fetch_model_classifications()
        existing_ids = self._fetch_human_tweet_ids()

        tweet_classifications: dict = defaultdict(list)
        for tweet_id, is_finance, sentiment in model_rows:
            tweet_classifications[tweet_id].append((is_finance, sentiment))

        rows_to_insert: List[Tuple] = []
        skipped = 0

        for tweet_id, entries in tweet_classifications.items():
            if tweet_id in existing_ids:
                skipped += 1
                continue

            finance_votes = sum(1 for is_f, _ in entries if is_f == 1)
            is_finance = 1 if finance_votes >= len(entries) / 2.0 else 0

            if is_finance == 0:
                sentiment = None
            else:
                candidates = [s for is_f, s in entries if is_f == 1 and s]
                sentiment = Counter(candidates).most_common(1)[0][0] if candidates else None

            rows_to_insert.append((tweet_id, is_finance, sentiment))

        inserted = self._bulk_insert_human_labels(rows_to_insert)
        logger.info(
            f"bootstrap_human_labels: {inserted} inseridos, {skipped} ignorados."
        )
        return {"inserted": inserted, "skipped": skipped}

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _fetch_model_classifications(self) -> List[Tuple]:
        """Retorna (tweet_id, is_finance_news, sentiment) de classificadores automáticos."""
        query = """
        SELECT tweet_id, is_finance_news, sentiment
        FROM tweets_classification
        WHERE classificator != 'Humano'
        ORDER BY tweet_id;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Falha ao carregar classificações automáticas: {e}")
            return []

    def _fetch_human_tweet_ids(self) -> set:
        """Retorna o conjunto de tweet_ids que já possuem classificação Humano."""
        query = """
        SELECT DISTINCT tweet_id
        FROM tweets_classification
        WHERE classificator = 'Humano';
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return {row[0] for row in cur.fetchall()}
        except Exception as e:
            logger.error(f"Falha ao carregar tweet_ids com classificação Humano: {e}")
            return set()

    def _bulk_insert_human_labels(self, rows: List[Tuple]) -> int:
        """Insere registros (tweet_id, is_finance_news, sentiment) com classificator='Humano'.

        Returns:
            Número de linhas efetivamente inseridas.
        """
        if not rows:
            return 0
        query = """
        INSERT INTO tweets_classification
            (tweet_id, is_finance_news, why_is_finance_news, sentiment, why_sentiment, classificator, score)
        VALUES (%s, %s, '', lower(%s), '', 'Humano', NULL)
        ON CONFLICT DO NOTHING;
        """
        params = [(tweet_id, is_finance, sentiment or "") for tweet_id, is_finance, sentiment in rows]
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(query, params)
                    inserted = cur.rowcount
                    conn.commit()
                    logger.debug(f"_bulk_insert_human_labels: {inserted} linhas inseridas.")
                    return inserted if inserted >= 0 else len(rows)
        except Exception as e:
            logger.error(f"Falha no bulk insert de labels Humano: {e}")
            conn.rollback()
            return 0

    def _delete_human_labels(self) -> None:
        """Remove todos os registros com classificator='Humano'."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM tweets_classification WHERE classificator = 'Humano';")
                    conn.commit()
                    logger.debug("_delete_human_labels: registros Humano removidos.")
        except Exception as e:
            logger.error(f"Falha ao remover labels Humano: {e}")
            conn.rollback()


if __name__ == "__main__":
    repo = ClassificationRepository()
    result = repo.bootstrap_human_labels(force=False)
    print(result)