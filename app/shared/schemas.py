from dataclasses import dataclass
from typing import List


@dataclass
class Annotation:
    """Representa uma anotação de entidade em um tweet."""

    def __init__(self, json: dict):
        self.start = json.get("start")
        self.end = json.get("end")
        self.probability = json.get("probability")
        self.type = json.get("type")
        self.normalized_text = json.get("normalized_text")

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "probability": self.probability,
            "type": self.type,
            "normalized_text": self.normalized_text,
        }


@dataclass
class Url:
    """Representa uma URL em um tweet."""

    def __init__(self, json: dict):
        self.start = json.get("start")
        self.end = json.get("end")
        self.url = json.get("url")
        self.expanded_url = json.get("expanded_url")
        self.display_url = json.get("display_url")
        self.media_key = json.get("media_key")

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "url": self.url,
            "expanded_url": self.expanded_url,
            "display_url": self.display_url,
            "media_key": self.media_key,
        }


@dataclass
class Entities:
    """Representa as entidades de um tweet (anotações e URLs)."""

    def __init__(self, json: dict):
        self.annotations = [Annotation(a) for a in json.get("annotations", [])]
        self.urls = [Url(u) for u in json.get("urls", [])]

    def to_dict(self):
        return {
            "annotations": [a.to_dict() for a in self.annotations],
            "urls": [u.to_dict() for u in self.urls],
        }


@dataclass
class PublicMetrics:
    """Representa as métricas públicas de um tweet."""

    def __init__(self, json: dict):
        self.like_count = json.get("like_count", 0)
        self.retweet_count = json.get("retweet_count", 0)
        self.reply_count = json.get("reply_count", 0)
        self.quote_count = json.get("quote_count", 0)

    def to_dict(self):
        return {
            "like_count": self.like_count,
            "retweet_count": self.retweet_count,
            "reply_count": self.reply_count,
            "quote_count": self.quote_count,
        }


@dataclass
class Hashtag:
    """Representa uma hashtag em um tweet."""

    def __init__(self, json: dict):
        self.start = json.get("start")
        self.end = json.get("end")
        self.tag = json.get("tag")

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "tag": self.tag,
        }


@dataclass
class NoteTweetEntities:
    """Representa as entidades de um note tweet (tweets longos)."""

    def __init__(self, json: dict):
        self.hashtags = [Hashtag(h) for h in json.get("hashtags", [])]

    def to_dict(self):
        return {"hashtags": [h.to_dict() for h in self.hashtags]}


@dataclass
class NoteTweet:
    """Representa o conteúdo estendido de um tweet longo."""

    def __init__(self, json: dict):
        self.text = json.get("text")
        self.entities = NoteTweetEntities(json.get("entities", {}))

    def to_dict(self):
        return {
            "text": self.text,
            "entities": self.entities.to_dict(),
        }


@dataclass
class Tweet:
    """Representa um tweet com todos os campos retornados pela API X v2."""

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
        self.id = json.get("id", "")
        self.text = json.get("text", "")
        self.author_id = json.get("author_id", "")
        self.lang = json.get("lang", "")
        self.created_at = json.get("created_at", "")
        self.entities = Entities(json.get("entities", {}))
        self.note_tweet = NoteTweet(json.get("note_tweet", {}))
        self.public_metrics = PublicMetrics(json.get("public_metrics", {}))
        self.edit_history_tweet_ids = json.get("edit_history_tweet_ids", [])

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "author_id": self.author_id,
            "lang": self.lang,
            "created_at": self.created_at,
            "entities": self.entities.to_dict(),
            "note_tweet": self.note_tweet.to_dict(),
            "public_metrics": self.public_metrics.to_dict(),
            "edit_history_tweet_ids": self.edit_history_tweet_ids,
        }


@dataclass
class Meta:
    """Representa os metadados de paginação da resposta da API X v2."""

    result_count: int
    newest_id: str
    oldest_id: str
    next_token: str

    def __init__(self, json: dict):
        self.result_count = json.get("result_count", 0)
        self.newest_id = json.get("newest_id", "")
        self.oldest_id = json.get("oldest_id", "")
        self.next_token = json.get("next_token", "")


@dataclass
class Tweets:
    """Representa a resposta completa da API X v2 para uma listagem de tweets."""

    data: List[Tweet]
    meta: Meta

    def __init__(self, json: dict):
        self.data = [Tweet(tweet) for tweet in json.get("data", [])]
        self.meta = Meta(json.get("meta", {}))