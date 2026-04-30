import pandas as pd
from psycopg2.extras import Json
from loguru import logger
from typing import Optional

from app.shared.database import DatabaseManager


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
        
    def query_classification_pairs(self, model_classificator: str) -> pd.DataFrame:
        """Retorna pares de classificação (Humano, Modelo) para o mesmo tweet.

        Usado pela avaliação para comparar o gold standard humano com
        as predições de um modelo específico.

        Args:
            model_classificator: Nome do classificador a comparar com 'Humano'.
                                Ex: 'FinBERT-PT-BR', 'SentiLex-PT', 'OpLexicon'.

        Returns:
            DataFrame com colunas tweet_id, human_label, model_label.
        """
        query = """
        SELECT
            h.tweet_id,
            h.sentiment AS human_label,
            m.sentiment AS model_label
        FROM tweets_classification h
        JOIN tweets_classification m ON h.tweet_id = m.tweet_id
        WHERE h.classificator = 'Humano'
        AND m.classificator = %s
        ORDER BY h.tweet_id;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (model_classificator,))
                    results = cur.fetchall()
                    logger.info(
                        f"Pares Humano vs {model_classificator}: {len(results)} encontrados."
                    )
                    return pd.DataFrame(
                        results,
                        columns=["tweet_id", "human_label", "model_label"],
                    )
        except Exception as e:
            logger.error(f"Falha ao buscar pares de classificação: {e}")
            return pd.DataFrame()