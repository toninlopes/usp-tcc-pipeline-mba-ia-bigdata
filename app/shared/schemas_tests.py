import pytest

from app.shared.schemas import (
    Annotation,
    Url,
    Entities,
    PublicMetrics,
    Hashtag,
    NoteTweetEntities,
    NoteTweet,
    Tweet,
    Meta,
    Tweets,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def annotation_dict():
    return {
        "start": 10,
        "end": 20,
        "probability": 0.95,
        "type": "Person",
        "normalized_text": "Petrobras",
    }


@pytest.fixture
def url_dict():
    return {
        "start": 30,
        "end": 53,
        "url": "https://t.co/abc123",
        "expanded_url": "https://infomoney.com.br/noticia",
        "display_url": "infomoney.com.br/noticia",
        "media_key": None,
    }


@pytest.fixture
def hashtag_dict():
    return {"start": 5, "end": 10, "tag": "IBOV"}


@pytest.fixture
def public_metrics_dict():
    return {
        "like_count": 42,
        "retweet_count": 7,
        "reply_count": 3,
        "quote_count": 1,
    }


@pytest.fixture
def note_tweet_dict(hashtag_dict):
    return {
        "text": "Mercado em alta nesta semana. #IBOV",
        "entities": {"hashtags": [hashtag_dict]},
    }


@pytest.fixture
def tweet_dict(url_dict, annotation_dict, public_metrics_dict, note_tweet_dict):
    return {
        "id": "1234567890",
        "text": "Mercado em alta nesta semana.",
        "author_id": "59773459",
        "lang": "pt",
        "created_at": "2025-11-01T12:00:00Z",
        "entities": {
            "annotations": [annotation_dict],
            "urls": [url_dict],
        },
        "note_tweet": note_tweet_dict,
        "public_metrics": public_metrics_dict,
        "edit_history_tweet_ids": ["1234567890"],
    }


@pytest.fixture
def meta_dict():
    return {
        "result_count": 10,
        "newest_id": "9999",
        "oldest_id": "1111",
        "next_token": "token_xyz",
    }


@pytest.fixture
def tweets_dict(tweet_dict, meta_dict):
    return {"data": [tweet_dict], "meta": meta_dict}


# ── Annotation ────────────────────────────────────────────────────────────────

class TestAnnotation:
    def test_parses_all_fields(self, annotation_dict):
        a = Annotation(annotation_dict)
        assert a.start == 10
        assert a.end == 20
        assert a.probability == 0.95
        assert a.type == "Person"
        assert a.normalized_text == "Petrobras"

    def test_to_dict_roundtrip(self, annotation_dict):
        assert Annotation(annotation_dict).to_dict() == annotation_dict

    def test_missing_fields_use_none(self):
        a = Annotation({})
        assert a.start is None
        assert a.normalized_text is None


# ── Url ───────────────────────────────────────────────────────────────────────

class TestUrl:
    def test_parses_all_fields(self, url_dict):
        u = Url(url_dict)
        assert u.url == "https://t.co/abc123"
        assert u.expanded_url == "https://infomoney.com.br/noticia"
        assert u.display_url == "infomoney.com.br/noticia"
        assert u.media_key is None

    def test_to_dict_roundtrip(self, url_dict):
        assert Url(url_dict).to_dict() == url_dict

    def test_missing_fields_use_none(self):
        u = Url({})
        assert u.url is None
        assert u.expanded_url is None


# ── Entities ──────────────────────────────────────────────────────────────────

class TestEntities:
    def test_parses_annotations_and_urls(self, annotation_dict, url_dict):
        e = Entities({"annotations": [annotation_dict], "urls": [url_dict]})
        assert len(e.annotations) == 1
        assert len(e.urls) == 1
        assert e.annotations[0].normalized_text == "Petrobras"
        assert e.urls[0].url == "https://t.co/abc123"

    def test_empty_lists_on_missing_keys(self):
        e = Entities({})
        assert e.annotations == []
        assert e.urls == []

    def test_to_dict_roundtrip(self, annotation_dict, url_dict):
        data = {"annotations": [annotation_dict], "urls": [url_dict]}
        assert Entities(data).to_dict() == data


# ── PublicMetrics ─────────────────────────────────────────────────────────────

class TestPublicMetrics:
    def test_parses_all_fields(self, public_metrics_dict):
        m = PublicMetrics(public_metrics_dict)
        assert m.like_count == 42
        assert m.retweet_count == 7
        assert m.reply_count == 3
        assert m.quote_count == 1

    def test_to_dict_roundtrip(self, public_metrics_dict):
        assert PublicMetrics(public_metrics_dict).to_dict() == public_metrics_dict

    def test_missing_fields_default_to_zero(self):
        m = PublicMetrics({})
        assert m.like_count == 0
        assert m.retweet_count == 0


# ── Hashtag ───────────────────────────────────────────────────────────────────

class TestHashtag:
    def test_parses_all_fields(self, hashtag_dict):
        h = Hashtag(hashtag_dict)
        assert h.start == 5
        assert h.end == 10
        assert h.tag == "IBOV"

    def test_to_dict_roundtrip(self, hashtag_dict):
        assert Hashtag(hashtag_dict).to_dict() == hashtag_dict

    def test_missing_fields_use_none(self):
        h = Hashtag({})
        assert h.tag is None


# ── NoteTweetEntities ─────────────────────────────────────────────────────────

class TestNoteTweetEntities:
    def test_parses_hashtags(self, hashtag_dict):
        nte = NoteTweetEntities({"hashtags": [hashtag_dict]})
        assert len(nte.hashtags) == 1
        assert nte.hashtags[0].tag == "IBOV"

    def test_empty_list_on_missing_key(self):
        assert NoteTweetEntities({}).hashtags == []

    def test_to_dict_roundtrip(self, hashtag_dict):
        data = {"hashtags": [hashtag_dict]}
        assert NoteTweetEntities(data).to_dict() == data


# ── NoteTweet ─────────────────────────────────────────────────────────────────

class TestNoteTweet:
    def test_parses_text_and_entities(self, note_tweet_dict):
        nt = NoteTweet(note_tweet_dict)
        assert nt.text == "Mercado em alta nesta semana. #IBOV"
        assert len(nt.entities.hashtags) == 1

    def test_to_dict_roundtrip(self, note_tweet_dict):
        assert NoteTweet(note_tweet_dict).to_dict() == note_tweet_dict

    def test_missing_text_is_none(self):
        assert NoteTweet({}).text is None


# ── Tweet ─────────────────────────────────────────────────────────────────────

class TestTweet:
    def test_parses_all_fields(self, tweet_dict):
        t = Tweet(tweet_dict)
        assert t.id == "1234567890"
        assert t.author_id == "59773459"
        assert t.lang == "pt"
        assert t.created_at == "2025-11-01T12:00:00Z"
        assert len(t.entities.annotations) == 1
        assert t.note_tweet.text == "Mercado em alta nesta semana. #IBOV"
        assert t.public_metrics.like_count == 42
        assert t.edit_history_tweet_ids == ["1234567890"]

    def test_to_dict_roundtrip(self, tweet_dict):
        assert Tweet(tweet_dict).to_dict() == tweet_dict

    def test_missing_fields_use_defaults(self):
        t = Tweet({})
        assert t.id == ""
        assert t.text == ""
        assert t.edit_history_tweet_ids == []


# ── Meta ──────────────────────────────────────────────────────────────────────

class TestMeta:
    def test_parses_all_fields(self, meta_dict):
        m = Meta(meta_dict)
        assert m.result_count == 10
        assert m.newest_id == "9999"
        assert m.oldest_id == "1111"
        assert m.next_token == "token_xyz"

    def test_missing_fields_use_defaults(self):
        m = Meta({})
        assert m.result_count == 0
        assert m.newest_id == ""
        assert m.next_token == ""


# ── Tweets ────────────────────────────────────────────────────────────────────

class TestTweets:
    def test_parses_data_and_meta(self, tweets_dict):
        tw = Tweets(tweets_dict)
        assert len(tw.data) == 1
        assert tw.data[0].id == "1234567890"
        assert tw.meta.result_count == 10
        assert tw.meta.next_token == "token_xyz"

    def test_empty_data_list(self):
        tw = Tweets({"data": [], "meta": {}})
        assert tw.data == []

    def test_missing_keys_use_defaults(self):
        tw = Tweets({})
        assert tw.data == []
        assert tw.meta.result_count == 0