import pandas as pd
from typing import Optional
from shared.database import DatabaseManager


class DatabaseTweetsQuery:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager if db_manager else DatabaseManager()

    def fetch_all_tweets(self) -> pd.DataFrame:
        """ "
        Fetch all tweets from the database and return as a DataFrame.
        """
        rows = self.db_manager.execute_query(
            """
            SELECT * FROM tweets t
            ORDER BY t.created_at DESC
            """
        )
        df = pd.DataFrame(
            rows,
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

        return df

    def fetch_tweets_with_human_classification(self) -> pd.DataFrame:
        """
        Fetch tweets that have human classification and return as a DataFrame.
        """
        rows = self.db_manager.execute_query(
            """
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
        )
        df = pd.DataFrame(
            rows,
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
            ],
        )
        return df
