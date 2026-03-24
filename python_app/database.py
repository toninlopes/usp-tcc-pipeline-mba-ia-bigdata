import psycopg2
from psycopg2.extras import Json
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import os
from loguru import logger
from dotenv import load_dotenv
import json
from parse_tweet import Tweets, Tweet

load_dotenv()


class DatabaseManager:
    """Database manager class to handle connection pooling and queries."""

    def __init__(self):
        self._pool = None

        self._db_name = os.getenv("POSTGRES_DB", "twitter_db")
        self._db_user = os.getenv(
            "POSTGRES_TWITTER_USER", os.getenv("POSTGRES_USER", "postgres")
        )
        self._db_password = os.getenv(
            "POSTGRES_TWITTER_PASSWORD", os.getenv("POSTGRES_PASSWORD", "")
        )
        self._db_host = os.getenv("POSTGRES_HOST", "localhost")
        self._db_port = int(os.getenv("POSTGRES_PORT", "5432"))

    def get_connection_params(self):
        """Gets the connection parameters as a dictionary."""
        return {
            "host": self._db_host,
            "port": self._db_port,
            "dbname": self._db_name,
            "user": self._db_user,
            "password": self._db_password,
        }

    def initialize_pool(self, minconn=1, maxconn=10):
        """Initializes the connection pool."""

        if self._pool is None:
            logger.info("Initializing database connection pool...")

            self._pool = SimpleConnectionPool(
                minconn, maxconn, **self.get_connection_params()
            )
            logger.info("Database connection pool initialized successfully.")

    def check_connection(self):
        """Checks if a connection can be established."""
        try:
            with psycopg2.connect(
                dbname=self._db_name,
                user=self._db_user,
                password=self._db_password,
                host=self._db_host,
                port=self._db_port,
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    result = cur.fetchone()
            print("Connection OK:", result)
            return 0
        except Exception as exc:
            print("Connection failed:", exc)
            return 1

    @contextmanager
    def get_connection(self):
        """Context manager to get a connection from the pool."""

        if self._pool is None:
            self.initialize_pool()  # Ensure the pool is initialized

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            self._pool.putconn(conn)

    def insert_tweet(self, tweet_data: dict) -> bool:
        """
        Inserts a single tweet into the database.

        Args:
            tweet_data (dict): Dictionary containing tweet data with keys:
                - tweet_id (str): Unique identifier of the tweet
                - username (str): Username of the tweet author
                - note_tweet (str): Text/content of the tweet
                - created_at (datetime): Creation timestamp of the tweet
                - likes (int): Number of likes the tweet received
                - hashtags (list): List of hashtags in the tweet
                - tweet (dict): Full tweet data as a dictionary

        Returns:
            bool: True if insertion was successful, False otherwise
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
                            (
                                Json(tweet_data.get("hashtags", []))
                                if tweet_data.get("hashtags")
                                else None
                            ),
                            (
                                Json(tweet_data.get("tweet", {}))
                                if tweet_data.get("tweet")
                                else None
                            ),
                        ),
                    )
                    conn.commit()
                    logger.info(
                        f"Tweet {tweet_data['tweet_id']} inserted successfully."
                    )
                    return True
        except Exception as e:
            logger.error(f"Failed to insert tweet {tweet_data['tweet_id']}: {e}")
            if conn:
                conn.rollback()
            return False

    def query_all_tweets(self) -> list:
        """
        Queries the tweets table for all entries.

        Args:
            is_finance_news (dict): The is_finance_news value to filter tweets by.
        """

        query = """
        SELECT * FROM tweets t
        ORDER BY t.created_at DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    logger.info(f"Queried all unlabeled tweets successfully.")
                    return results
        except Exception as e:
            logger.error(f"Failed to query all unlabeled tweets: {e}")
            return []

    def query_all_tweets_with_human_classification(self) -> list:
        """
        Queries the tweets table for all entries.

        Args:
            is_finance_news (dict): The is_finance_news value to filter tweets by.
        """

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
                    logger.info(f"Queried all unlabeled tweets successfully.")
                    return results
        except Exception as e:
            logger.error(f"Failed to query all unlabeled tweets: {e}")
            return []

    def update_sentiment(self, id: int, sentiment: str) -> bool:
        """
        Updates the sentiment of a tweet in the database.

        Args:
            id (int): The ID of the tweet to update.
            sentiment (str): The sentiment label to set (e.g., 'positivo', 'negativo', 'neutro').

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        query = "UPDATE tweets SET sentiment = lower(%s) WHERE id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (sentiment, id))
                    conn.commit()
                    logger.info(f"Sentiment for tweet ID {id} updated successfully.")
                    return True
        except Exception as e:
            logger.error(f"Failed to update sentiment for tweet ID {id}: {e}")
            if conn:
                conn.rollback()
            return False

    def update_is_finance_news(self, id: int, is_finance_news: int) -> bool:
        """
        Updates the is_finance_news field of a tweet in the database.

        Args:
            id (int): The ID of the tweet to update.
            is_finance_news (int): The is_finance_news value to set (0 or 1).

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        query = "UPDATE tweets SET is_finance_tweet = %s WHERE id = %s"
        if is_finance_news == 0:
            query = "UPDATE tweets SET is_finance_tweet = %s, sentiment = NULL WHERE id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (0 if not is_finance_news else 1, id))
                    conn.commit()
                    logger.info(
                        f"Is finance news for tweet ID {id} updated successfully."
                    )
                    return True
        except Exception as e:
            logger.error(f"Failed to update is_finance_news for tweet ID {id}: {e}")
            if conn:
                conn.rollback()
            return False

    def query_tweets_classification_by_id(self, tweet_id: int) -> list:
        """
        Queries the tweets_classification table for entries with a specific tweet_id.

        Args:
            tweet_id (int): The tweet_id to filter tweets by.
        """

        # 1. Force conversion to a standard Python int immediately
        # If tweet_id is passed as [12345] or a numpy array, .item() extracts it.
        if hasattr(tweet_id, "item"):
            clean_id = int(tweet_id.item())  # type: ignore
        elif isinstance(tweet_id, (list, tuple)):
            clean_id = int(tweet_id[0])
        else:
            clean_id = int(tweet_id)

        query = """
        SELECT * FROM tweets_classification
        WHERE tweet_id = %s
        ORDER BY id DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (clean_id,))
                    results = cur.fetchall()
                    logger.info(
                        f"Queried tweets classification for tweet ID {clean_id} successfully."
                    )
                    return results
        except Exception as e:
            logger.error(
                f"Failed to query tweets classification for tweet ID {clean_id}: {e}"
            )
            return []

    def insert_tweets_classification(
        self,
        tweet_id: int,
        is_finance_news: int,
        why_is_finance_news: str,
        sentiment: str,
        why_sentiment: str,
        classificator: str,
    ) -> bool:
        """
        Inserts a classification entry into the tweets_classification table.

        Args:
            tweet_id (int): The tweet_id to insert.
            is_finance_news (int): The is_finance_news value to set (0 or 1).
            why_is_finance_news (str): The reason for the is_finance_news value.
            sentiment (str): The sentiment value.
            why_sentiment (str): The reason for the sentiment value.
            classificater (str): The name of the person who classified the tweet.
        """
        # 1. Force conversion to a standard Python int immediately
        # If tweet_id is passed as [12345] or a numpy array, .item() extracts it.
        if hasattr(tweet_id, "item"):
            clean_id = int(tweet_id.item())  # type: ignore
        elif isinstance(tweet_id, (list, tuple)):
            clean_id = int(tweet_id[0])
        else:
            clean_id = int(tweet_id)

        clean_is_finance_news = int(is_finance_news)
        query = """
        INSERT INTO tweets_classification (
            tweet_id,
            is_finance_news,
            why_is_finance_news,
            sentiment,
            why_sentiment,
            classificator
        ) VALUES (%s, %s, %s, lower(%s), %s, %s)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            clean_id,
                            clean_is_finance_news,
                            why_is_finance_news,
                            sentiment,
                            why_sentiment,
                            classificator,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        f"Inserted tweets classification for tweet ID {clean_id} successfully."
                    )
                    return True
        except Exception as e:
            logger.error(
                f"Failed to insert tweets classification for tweet ID {clean_id}: {e}"
            )
            if conn:
                conn.rollback()
            return False

    def query_log(self, search_term: dict) -> list:
        """
        Queries the collection_log table for entries matching the given search term.

        Args:
            search_term (dict): The search term to filter logs by.
        """

        query = """
        SELECT * FROM collection_log
        WHERE search_term @> %s::jsonb
        ORDER BY start_time DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    jsonb = Json(search_term) if search_term else None
                    cur.execute(query, (jsonb,))
                    results = cur.fetchall()
                    logger.info(
                        f"Queried collection log for search term '{search_term}' successfully."
                    )
                    return results
        except Exception as e:
            logger.error(
                f"Failed to query collection log for search term '{search_term}': {e}"
            )
            return []

    def query_all_pending_logs(self) -> list:
        """
        Queries the collection_log table for all entries with status 'pending'.
        """

        query = """
        SELECT * FROM collection_log
        WHERE status = 'pending'
        ORDER BY (search_term->>'to_date_time') DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    logger.info("Queried all pending collection logs successfully.")
                    return results
        except Exception as e:
            logger.error(f"Failed to query pending collection logs: {e}")
            return []

    def insert_log(self, log_data: dict) -> bool:
        """
        Inserts a log entry into the collection_log table.

        Args:
            log_data (dict): Dictionary containing log data with keys:
                - search_term (dict): Search term used for the collection
                - tweets_collected (int): Number of tweets collected
                - start_time (datetime): Start time of the collection
                - end_time (datetime): End time of the collection
                - status (str): Status of the collection (e.g., 'pending', 'collected')
                - error_message (str): Error message if any issues occurred during collection

        Returns:
            bool: True if insertion was successful, False otherwise
        """

        query = """
        INSERT INTO collection_log (search_term, tweets_collected, start_time, end_time, status, error_message)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    jsonb = (
                        Json(log_data.get("search_term", {}))
                        if log_data.get("search_term")
                        else None
                    )

                    cur.execute(
                        query,
                        (
                            jsonb,
                            log_data["tweets_collected"],
                            log_data["start_time"],
                            log_data["end_time"],
                            log_data["status"],
                            log_data.get("error_message"),
                        ),
                    )
                    conn.commit()
                    logger.info("Collection log inserted successfully.")
                    return True
        except Exception as e:
            logger.error(f"Failed to insert collection log: {e}")
            if conn:
                conn.rollback()
            return False

    def update_log(self, log_id: int, update_data: dict) -> bool:
        """
        Updates a log entry in the collection_log table.

        Args:
            log_id (int): The ID of the log entry to update.
            update_data (dict): Dictionary containing the fields to update with their new values.
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        set_clause = ", ".join([f"{key} = %s" for key in update_data.keys()])
        query = f"UPDATE collection_log SET {set_clause} WHERE id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    values = list(update_data.values()) + [log_id]
                    cur.execute(query, values)
                    conn.commit()
                    logger.info(
                        f"Collection log with ID {log_id} updated successfully."
                    )
                    return True
        except Exception as e:
            logger.error(f"Failed to update collection log with ID {log_id}: {e}")
            if conn:
                conn.rollback()
            return False

    def close_pool(self):
        """Closes all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed.")


if __name__ == "__main__":
    db_manager = DatabaseManager()
    result = db_manager.check_connection()
    if result == 0:
        print("Database connection successful.")
    else:
        print("Database connection failed.")

    try:
        # Caminho do script atual
        script_dir = os.path.dirname(__file__)  # pasta collect-twitter-data/python_app
        print(f"Diretório do script: {script_dir}")

        # Subir 2 níveis: collect-twitter-data/python_app → collect-twitter-data
        project_root = os.path.dirname(script_dir)
        print(f"Raiz do projeto: {project_root}")

        # Caminho completo para o JSON
        json_path = os.path.join(project_root, "json_data", "x_tweets.json")
        print(f"Caminho do JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
            # print(json_data)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    tweets = Tweets(json_data)
    tweets_not_inserted = list[Tweet]()
    for tweet in tweets.data:
        if (
            tweet.note_tweet is None
            or tweet.note_tweet.text is None
            or tweet.text is None
        ):
            logger.warning(f"Tweet {tweet.id} has no text content. Skipping insertion.")
            continue

        query = f"INSERT INTO tweets (tweet_id, username, note_tweet, created_at, likes, hashtags, tweet) \
        VALUES (%s, %s, %s, %s, %s, %s, %s);"

        inserted = db_manager.insert_tweet(
            {
                "tweet_id": tweet.id,
                "username": tweet.author_id,
                "note_tweet": tweet.note_tweet.text if tweet.note_tweet else tweet.text,
                "created_at": tweet.created_at,
                "likes": tweet.public_metrics.like_count,
                "hashtags": (
                    tweet.note_tweet.entities.to_dict()
                    if tweet.note_tweet
                    and tweet.note_tweet.entities
                    and tweet.note_tweet.entities.hashtags
                    else None
                ),
                "tweet": tweet.to_dict(),
            }
        )

        if not inserted:
            tweets_not_inserted.append(tweet)

    if len(tweets_not_inserted) > 0:
        logger.warning(
            f"{len(tweets_not_inserted)} tweets were not inserted into the database."
        )
    else:
        logger.info("All tweets were inserted successfully.")
