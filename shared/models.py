from dataclasses import dataclass
from typing import List
import os
import json

@dataclass
class Annotation:
    """Class to represent an annotation in a tweet."""
    
    def __init__(self, json: dict):
        self.start = json.get('start')
        self.end = json.get('end')
        self.probability = json.get('probability')
        self.type = json.get('type')
        self.normalized_text = json.get('normalized_text')

    def to_dict(self):
        """Converts the Annotation to a dictionary format."""
        return {
            'start': self.start,
            'end': self.end,
            'probability': self.probability,
            'type': self.type,
            'normalized_text': self.normalized_text
        }

@dataclass
class Url:
    """Class to represent a URL in a tweet."""
    
    def __init__(self, json: dict):
        self.start = json.get('start')
        self.end = json.get('end')
        self.url = json.get('url')
        self.expanded_url = json.get('expanded_url')
        self.display_url = json.get('display_url')
        self.media_key = json.get('media_key')

    def to_dict(self):
        """Converts the Url to a dictionary format."""
        return {
            'start': self.start,
            'end': self.end,
            'url': self.url,
            'expanded_url': self.expanded_url,
            'display_url': self.display_url,
            'media_key': self.media_key
        }

@dataclass
class Entities:
    """Class to represent the entities of a tweet."""
    
    def __init__(self, json: dict):
        self.annotations =  [Annotation(annotation) for annotation in json.get('annotations', [])]
        self.urls = [Url(url) for url in json.get('urls', [])]

    def to_dict(self):
        """Converts the Entities to a dictionary format."""
        return {
            'annotations': [annotation.to_dict() for annotation in self.annotations],
            'urls': [url.to_dict() for url in self.urls]
        }

class PublicMetrics:
    """Class to represent public metrics of a tweet."""
    
    def __init__(self, json: dict):
        self.like_count = json.get('like_count', 0)
        self.retweet_count = json.get('retweet_count', 0)
        self.reply_count = json.get('reply_count', 0)
        self.quote_count = json.get('quote_count', 0)

    def to_dict(self):
        """Converts the PublicMetrics to a dictionary format."""
        return {
            'like_count': self.like_count,
            'retweet_count': self.retweet_count,
            'reply_count': self.reply_count,
            'quote_count': self.quote_count
        }

@dataclass
class Hashtag:
    """Class to represent a hashtag in a tweet."""
    
    def __init__(self, json: dict):
        self.start = json.get('start')
        self.end = json.get('end')
        self.tag = json.get('tag')

    def to_dict(self):
        """Converts the Hashtag to a dictionary format."""
        return {
            'start': self.start,
            'end': self.end,
            'tag': self.tag
        }

    def to_json(self):
        """Converts the Hashtag to a JSON-serializable format."""
        return json.dumps(self.to_dict())

@dataclass
class NoteTweetEntities:
    """Class to represent the entities of a note tweet."""
    
    def __init__(self, json: dict):
        self.hashtags = [Hashtag(hashtag) for hashtag in json.get('hashtags', [])]

    def to_dict(self):
        """Converts the NoteTweetEntities to a dictionary format."""
        return {
            'hashtags': [hashtag.to_dict() for hashtag in self.hashtags]
        }

    def to_json(self):
        """Converts the NoteTweetEntities to a JSON-serializable format."""
        return json.dumps(self.to_dict())

@dataclass
class NoteTweet:
    """Class to represent a note tweet and extract relevant information."""
    
    def __init__(self, json: dict):
        self.text = json.get('text')
        self.entities = NoteTweetEntities(json.get('entities', {}))

    def to_dict(self):
        """Converts the NoteTweet to a dictionary format."""
        return {
            'text': self.text,
            'entities': self.entities.to_dict()
        }
    
@dataclass
class Tweet:
    """Class to represent a tweet and extract relevant information."""
    
    id: str
    text: str
    author_id: str
    lang: str
    created_at: str
    entities: Entities
    note_tweet: NoteTweet
    public_metrics: PublicMetrics
    edit_history_tweet_ids: list

    def __init__(self, json: dict):
        self.id = json.get('id', '')
        self.text = json.get('text', '')
        self.author_id = json.get('author_id', '')
        self.lang = json.get('lang', '')
        self.created_at = json.get('created_at', '')
        self.entities = Entities(json.get('entities', {}))
        self.note_tweet = NoteTweet(json.get('note_tweet', {}))
        self.public_metrics = PublicMetrics(json.get('public_metrics', {}))
        self.edit_history_tweet_ids = json.get('edit_history_tweet_ids', [])
    
    def to_dict(self):
        """Converts the Tweet to a dictionary format."""
        return {
            'id': self.id,
            'text': self.text,
            'author_id': self.author_id,
            'lang': self.lang,
            'created_at': self.created_at,
            'entities': self.entities.to_dict(),
            'note_tweet': self.note_tweet.to_dict(),
            'public_metrics': self.public_metrics.to_dict(),
            'edit_history_tweet_ids': self.edit_history_tweet_ids
        }
    
@dataclass
class Meta:
    result_count: int
    newest_id: str
    oldest_id: str
    next_token: str

    def __init__(self, json: dict):
        self.result_count = json.get('result_count', 0)
        self.newest_id = json.get('newest_id', '')
        self.oldest_id = json.get('oldest_id', '')
        self.next_token = json.get('next_token', '')


@dataclass
class Tweets:
    """Class to parse tweet data and extract relevant information."""
    
    data: List[Tweet]
    meta: Meta
    
    def __init__(self, json: dict):
        self.data = [Tweet(tweet) for tweet in json.get('data', [])]
        self.meta = Meta(json.get('meta', {}))
    

if __name__ == "__main__":

    try:
        # Caminho do script atual
        script_dir = os.path.dirname(__file__)  # pasta python_app/parse_tweet/
        print(f"Diretório do script: {script_dir}")

        # Subir 2 níveis: python_app/parse_tweet/ → python_app/ → projeto/
        project_root = os.path.dirname(os.path.dirname(script_dir))
        print(f"Raiz do projeto: {project_root}")

        # Caminho completo para o JSON
        json_path = os.path.join(project_root, "config", "x_tweets.json")
        print(f"Caminho do JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        # print(json_data)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    # parser = ParseTweet()
    tweets = Tweets(json_data)
    for tweet in tweets.data:
        print("-" * 50)
        # print(f"Tweet ID: {tweet.id}")
        # print(f"Created at: {tweet.created_at}")

        # for annotation in tweet.entities.annotations:
        #     print(f"Annotation: {annotation.normalized_text} (Type: {annotation.type}, Probability: {annotation.probability})")
        # for url in tweet.entities.urls:
        #     print(f"URL: {url.url} (Expanded: {url.expanded_url}, Display: {url.display_url})")

        # print(f"Public Metrics: Likes: {tweet.public_metrics.like_count}, Retweets: {tweet.public_metrics.retweet_count}, Replies: {tweet.public_metrics.reply_count}, Quotes: {tweet.public_metrics.quote_count}")

        print(tweet.note_tweet.entities.to_json())
