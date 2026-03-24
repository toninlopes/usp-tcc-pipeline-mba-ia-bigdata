from database import DatabaseManager
from collection_log import SearchTerm
from parse_tweet import Tweet, Tweets
from x_tweets import TweetCollector
import pytz
from datetime import datetime

sao_paulo_tz = pytz.timezone('America/Sao_Paulo')

if __name__ == "__main__":
    db_manager = DatabaseManager()
    result = db_manager.check_connection()

    if result == 1:
        exit("Database connection failed. Exiting.")

    pending_logs = db_manager.query_all_pending_logs();
    for log in pending_logs:
        search_term = SearchTerm(log[1])  # Convertendo JSON para SearchTerm

        # Get current time in São Paulo
        now_sao_paulo = datetime.now(sao_paulo_tz)

        # Convert to UTC and format as ISO 8601 with 'Z' suffix
        utc_time = now_sao_paulo.astimezone(pytz.UTC)
        formatted_time = utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        db_manager.update_log(log[0], {'start_time': formatted_time})  # Atualizando status para 'in_progress'
        
        total_tweets_inserted = 0
        tweets_not_inserted = list[Tweet]()
        next_token = ''        
        while True:
            collector = TweetCollector(search_term.x_user_id, search_term.from_date_time, search_term.to_date_time, next_token)
            tweets_data = collector.make_request()
            tweets = Tweets(tweets_data)

            for tweet in tweets.data:
                if tweet.note_tweet is None or tweet.note_tweet.text is None or tweet.text is None:
                    print(f"Tweet {tweet.id} has no text content. Skipping insertion.")
                    continue # Pulando tweets que são apenas notas

                query = """
                INSERT INTO tweets (tweet_id, username, note_tweet, created_at, likes, hashtags, tweet)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """

                inserted = db_manager.insert_tweet({
                    'tweet_id': tweet.id,
                    'username': tweet.author_id,
                    'note_tweet': tweet.note_tweet.text if tweet.note_tweet else tweet.text,
                    'created_at': tweet.created_at,
                    'likes': tweet.public_metrics.like_count,
                    'hashtags': tweet.note_tweet.entities.to_dict() if tweet.note_tweet and tweet.note_tweet.entities and tweet.note_tweet.entities.hashtags else None,
                    'tweet': tweet.to_dict()
                })

                if not inserted:
                    tweets_not_inserted.append(tweet)

                total_tweets_inserted += 1

            if tweets.meta.next_token:
                next_token = tweets.meta.next_token
            else:
                next_token = ''
                break
        
        utc_time = now_sao_paulo.astimezone(pytz.UTC)
        formatted_time = utc_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        db_manager.update_log(log[0], {
            'end_time': formatted_time,
            'tweets_collected': total_tweets_inserted,
            'status': 'completed' if len(tweets_not_inserted) == 0 and total_tweets_inserted > 0 else 'partially_completed'
        })  # Atualizando status para 'in_progress'