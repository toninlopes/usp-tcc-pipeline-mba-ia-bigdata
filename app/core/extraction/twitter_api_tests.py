from unittest.mock import MagicMock, patch
import pytest

from app.core.extraction.twitter_api import TwitterAPIExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def extractor():
    with patch("app.core.extraction.twitter_api.load_dotenv"), \
         patch.dict("os.environ", {"TWITTER_ACCESS_TOKEN": "test_token"}, clear=True):
        return TwitterAPIExtractor(
            x_user_id="59773459",
            from_date_time="2025-10-01T00:00:00Z",
            to_date_time="2025-10-31T23:59:59Z",
        )


@pytest.fixture
def mock_response():
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [{"id": "1", "text": "tweet de teste"}],
        "meta": {"result_count": 1, "newest_id": "1", "oldest_id": "1"},
    }
    return response


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestInit:
    def test_builds_correct_base_url(self, extractor):
        assert extractor._base_url == "https://api.x.com/2/users/59773459/tweets"

    def test_sets_authorization_header(self, extractor):
        assert extractor._headers == {"Authorization": "Bearer test_token"}

    def test_sets_start_and_end_time(self, extractor):
        assert extractor._params["start_time"] == "2025-10-01T00:00:00Z"
        assert extractor._params["end_time"] == "2025-10-31T23:59:59Z"

    def test_sets_max_results_to_100(self, extractor):
        assert extractor._params["max_results"] == 100

    def test_does_not_include_pagination_token_by_default(self, extractor):
        assert "pagination_token" not in extractor._params

    def test_includes_pagination_token_when_provided(self):
        with patch("app.core.extraction.twitter_api.load_dotenv"), \
             patch.dict("os.environ", {"TWITTER_ACCESS_TOKEN": "test_token"}, clear=True):
            extractor = TwitterAPIExtractor(
                x_user_id="59773459",
                from_date_time="2025-10-01T00:00:00Z",
                to_date_time="2025-10-31T23:59:59Z",
                next_token="next_page_abc",
            )
        assert extractor._params["pagination_token"] == "next_page_abc"

    def test_empty_token_when_env_var_missing(self):
        with patch("app.core.extraction.twitter_api.load_dotenv"), \
             patch.dict("os.environ", {}, clear=True):
            extractor = TwitterAPIExtractor("59773459", "2025-10-01T00:00:00Z", "2025-10-31T23:59:59Z")
        assert extractor._bearer_token == ""
        assert extractor._headers == {"Authorization": "Bearer "}


# ── make_request ──────────────────────────────────────────────────────────────

class TestMakeRequest:
    def test_returns_json_on_success(self, extractor, mock_response):
        with patch("app.core.extraction.twitter_api.requests.get", return_value=mock_response):
            result = extractor.make_request()
        assert result["meta"]["result_count"] == 1
        assert result["data"][0]["id"] == "1"

    def test_calls_correct_url(self, extractor, mock_response):
        with patch("app.core.extraction.twitter_api.requests.get", return_value=mock_response) as mock_get:
            extractor.make_request()
        mock_get.assert_called_once_with(
            url=extractor._base_url,
            headers=extractor._headers,
            params=extractor._params,
        )

    def test_raises_on_non_200_status(self, extractor):
        error_response = MagicMock()
        error_response.status_code = 401
        error_response.text = "Unauthorized"
        with patch("app.core.extraction.twitter_api.requests.get", return_value=error_response):
            with pytest.raises(Exception, match="401"):
                extractor.make_request()

    def test_raises_on_500_status(self, extractor):
        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal Server Error"
        with patch("app.core.extraction.twitter_api.requests.get", return_value=error_response):
            with pytest.raises(Exception, match="500"):
                extractor.make_request()

    def test_raises_on_429_rate_limit(self, extractor):
        error_response = MagicMock()
        error_response.status_code = 429
        error_response.text = "Too Many Requests"
        with patch("app.core.extraction.twitter_api.requests.get", return_value=error_response):
            with pytest.raises(Exception, match="429"):
                extractor.make_request()


# ── fetch ─────────────────────────────────────────────────────────────────────

class TestFetch:
    def test_returns_list(self, extractor, mock_response):
        with patch("app.core.extraction.twitter_api.requests.get", return_value=mock_response):
            result = extractor.fetch("59773459", "2025-10-01T00:00:00Z", "2025-10-31T23:59:59Z")
        assert isinstance(result, list)

    def test_returns_list_with_one_element(self, extractor, mock_response):
        with patch("app.core.extraction.twitter_api.requests.get", return_value=mock_response):
            result = extractor.fetch("59773459", "2025-10-01T00:00:00Z", "2025-10-31T23:59:59Z")
        assert len(result) == 1

    def test_result_contains_response_json(self, extractor, mock_response):
        with patch("app.core.extraction.twitter_api.requests.get", return_value=mock_response):
            result = extractor.fetch("59773459", "2025-10-01T00:00:00Z", "2025-10-31T23:59:59Z")
        assert result[0]["data"][0]["id"] == "1"