import pandas as pd
from typing import Optional
from psycopg2.extras import Json
from loguru import logger

from app.shared.database import DatabaseManager

class TweetsRepository(DatabaseManager):
    """Queries e operações sobre a tabela `tweets`."""

    def insert_tweet(self, tweet_data: dict) -> bool:
        """Insere um único tweet no banco de dados.

        Args:
            tweet_data: Dicionário com as chaves tweet_id, username, note_tweet,
                        created_at, likes, hashtags, tweet.

        Returns:
            True se inserido com sucesso, False caso contrário.
        """
        query = """
        INSERT INTO tweets (tweet_id, username, note_tweet, created_at, likes, hashtags, tweet)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            tweet_data["tweet_id"],
                            tweet_data["username"],
                            tweet_data["note_tweet"],
                            tweet_data["created_at"],
                            tweet_data["likes"],
                            Json(tweet_data["hashtags"]) if tweet_data.get("hashtags") else None,
                            Json(tweet_data["tweet"]) if tweet_data.get("tweet") else None,
                        ),
                    )
                    conn.commit()
                    logger.info(f"Tweet {tweet_data['tweet_id']} inserido com sucesso.")
                    return True
        except Exception as e:
            logger.error(f"Falha ao inserir tweet {tweet_data['tweet_id']}: {e}")
            conn.rollback()
            return False

    def query_all_tweets(self) -> pd.DataFrame:
        """Retorna todos os tweets ordenados por data de publicação (mais recente primeiro)."""
        query = """
        SELECT * FROM tweets t
        ORDER BY t.created_at DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    logger.info("Consulta de todos os tweets realizada com sucesso.")
                    return pd.DataFrame(
                        results,
                        columns=[
                            "id",
                            "tweet_id",
                            "username",
                            "note_tweet",
                            "created_at",
                            "likes",
                            "hashtags",
                            "tweet",
                            "sentiment",
                            "is_finance_tweet",
                        ],
                    )
        except Exception as e:
            logger.error(f"Falha ao consultar tweets: {e}")
            return pd.DataFrame()

    def query_all_tweets_with_human_classification(self) -> pd.DataFrame:
        """Retorna todos os tweets indicando se possuem classificação humana."""
        query = """
        SELECT
            t.*,
            EXISTS (
                SELECT 1 FROM tweets_classification tc
                WHERE tc.tweet_id = t.id
                AND tc.classificator = 'Humano'
            ) AS has_human_classification
        FROM tweets t
        ORDER BY t.created_at DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    logger.info("Consulta de tweets com classificação humana realizada com sucesso.")
                    return pd.DataFrame(
                        results,
                        columns=[
                            "id",
                            "tweet_id",
                            "username",
                            "note_tweet",
                            "created_at",
                            "likes",
                            "hashtags",
                            "tweet",
                            "sentiment",
                            "is_finance_tweet",
                            "has_human_classification",
                        ],)
        except Exception as e:
            logger.error(f"Falha ao consultar tweets com classificação humana: {e}")
            return pd.DataFrame()

    def query_all_tweets_with_human_but_not_finbert_classification(self) -> pd.DataFrame:
        """Retorna tweets financeiros com classificação humana mas sem classificação FinBERT-PT-BR.

        Usado pelo módulo de processamento para determinar quais tweets ainda
        precisam ser classificados pelo modelo.
        """
        query = """
        WITH classified AS (
            SELECT
                t.*,
                EXISTS (
                    SELECT 1 FROM tweets_classification tc
                    WHERE tc.tweet_id = t.id
                    AND tc.classificator = 'Humano'
                ) AS has_human_classification,
                EXISTS (
                    SELECT 1 FROM tweets_classification tc
                    WHERE tc.tweet_id = t.id
                    AND tc.classificator = 'FinBERT-PT-BR'
                ) AS has_finbert_classification
            FROM tweets t
        )
        SELECT t.* FROM classified t
        WHERE
            is_finance_tweet = 1
            AND has_human_classification = TRUE
            AND has_finbert_classification = FALSE
        ORDER BY t.created_at DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    logger.info("Consulta de tweets sem classificação FinBERT realizada com sucesso.")
                    return pd.DataFrame(
                        results,
                        columns=[
                            "id",
                            "tweet_id",
                            "username",
                            "note_tweet",
                            "created_at",
                            "likes",
                            "hashtags",
                            "tweet",
                            "sentiment",
                            "is_finance_tweet",
                            "has_human_classification",
                            "has_finbert_classification",
                        ],
                    )
        except Exception as e:
            logger.error(f"Falha ao consultar tweets sem classificação FinBERT: {e}")
            return pd.DataFrame()

    def update_sentiment(self, id: int, sentiment: str) -> bool:
        """Atualiza o campo `sentiment` de um tweet na tabela `tweets`.

        Usado pela interface de anotação manual como atalho rápido,
        antes da classificação completa ser salva em `tweets_classification`.

        Args:
            id: Chave primária do tweet (tweets.id).
            sentiment: Rótulo de sentimento ('positivo', 'negativo', 'neutro').
        """
        query = "UPDATE tweets SET sentiment = lower(%s) WHERE id = %s"
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (sentiment, id))
                    conn.commit()
                    logger.info(f"Sentimento do tweet ID {id} atualizado com sucesso.")
                    return True
        except Exception as e:
            logger.error(f"Falha ao atualizar sentimento do tweet ID {id}: {e}")
            conn.rollback()
            return False

    def update_is_finance_news(self, id: int, is_finance_news: int) -> bool:
        """Atualiza o campo `is_finance_tweet` de um tweet na tabela `tweets`.

        Quando is_finance_news=0, o sentiment é zerado junto — um tweet
        não-financeiro não deve ter sentimento classificado.

        Args:
            id: Chave primária do tweet (tweets.id).
            is_finance_news: 1 para financeiro, 0 para não-financeiro.
        """
        if is_finance_news == 0:
            query = "UPDATE tweets SET is_finance_tweet = %s, sentiment = NULL WHERE id = %s"
        else:
            query = "UPDATE tweets SET is_finance_tweet = %s WHERE id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (is_finance_news, id))
                    conn.commit()
                    logger.info(f"Campo is_finance_tweet do tweet ID {id} atualizado com sucesso.")
                    return True
        except Exception as e:
            logger.error(f"Falha ao atualizar is_finance_tweet do tweet ID {id}: {e}")
            conn.rollback()
            return False