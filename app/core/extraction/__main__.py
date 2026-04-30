from typing import Optional
from datetime import datetime
from pathlib import Path

import pytz

from app.shared.db_collection_log import CollectionLogRepository
from app.shared.db_tweets import TweetsRepository
from app.shared.schemas import Tweets
from app.core.extraction.collection_log import SearchTerm
from app.core.extraction.twitter_api import TwitterAPIExtractor

SAO_PAULO_TZ = pytz.timezone("America/Sao_Paulo")


def now_utc_str() -> str:
    """Retorna o timestamp atual em UTC no formato ISO 8601."""
    return (
        datetime.now(SAO_PAULO_TZ)
        .astimezone(pytz.UTC)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# assinatura corrigida
def run(
    log_repo: Optional[CollectionLogRepository] = None,
    tweet_repo: Optional[TweetsRepository] = None,
) -> None:
    """Orquestra a coleta de tweets para todos os logs pendentes.

    Args:
        log_repo: Repositório de logs. Instanciado automaticamente se não fornecido.
        tweet_repo: Repositório de tweets. Instanciado automaticamente se não fornecido.
    """
    log_repo = log_repo or CollectionLogRepository()
    tweet_repo = tweet_repo or TweetsRepository()

    if log_repo.check_connection() == 1:
        raise SystemExit("Falha na conexão com o banco. Encerrando.")

    pending_logs = log_repo.query_all_pending_logs()

    if pending_logs.empty:
        print("Nenhum log pendente encontrado.")
        return

    for log in pending_logs.itertuples():
        search_term = SearchTerm(log.search_term)
        log_repo.update_log(log.id, {"start_time": now_utc_str()})

        total_inserted = 0
        total_failed = 0
        next_token = ""

        while True:
            collector = TwitterAPIExtractor(
                search_term.x_user_id,
                search_term.from_date_time,
                search_term.to_date_time,
                next_token,
            )
            tweets_data = collector.make_request()
            tweets = Tweets(tweets_data)

            for tweet in tweets.data:
                if not tweet.note_tweet or not tweet.note_tweet.text:
                    print(f"Tweet {tweet.id} sem conteúdo de texto. Ignorando.")
                    continue

                inserted = tweet_repo.insert_tweet({
                    "tweet_id": tweet.id,
                    "username": tweet.author_id,
                    "note_tweet": tweet.note_tweet.text,
                    "created_at": tweet.created_at,
                    "likes": tweet.public_metrics.like_count,
                    "hashtags": (
                        tweet.note_tweet.entities.to_dict()
                        if tweet.note_tweet.entities.hashtags
                        else None
                    ),
                    "tweet": tweet.to_dict(),
                })

                if inserted:
                    total_inserted += 1
                else:
                    total_failed += 1

            if tweets.meta.next_token:
                next_token = tweets.meta.next_token
            else:
                break

        status = (
            "completed"
            if total_failed == 0 and total_inserted > 0
            else "partially_completed"
        )
        log_repo.update_log(log.id, {
            "end_time": now_utc_str(),
            "tweets_collected": total_inserted,
            "status": status,
        })

        print(f"Coleta finalizada: {total_inserted} inseridos, {total_failed} falhas — {status}")


if __name__ == "__main__":
    run()