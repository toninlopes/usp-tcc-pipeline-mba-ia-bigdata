import json
import pytest

from app.core.extraction.collection_log import SearchTerm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def search_term_dict():
    return {
        "x_user_id": "59773459",
        "from_date_time": "2025-10-01T00:00:00Z",
        "to_date_time": "2025-10-31T23:59:59Z",
    }


@pytest.fixture
def search_term(search_term_dict):
    return SearchTerm(search_term_dict)


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestSearchTermInit:
    def test_parses_all_fields(self, search_term):
        assert search_term.x_user_id == "59773459"
        assert search_term.from_date_time == "2025-10-01T00:00:00Z"
        assert search_term.to_date_time == "2025-10-31T23:59:59Z"

    def test_missing_fields_default_to_empty_string(self):
        term = SearchTerm({})
        assert term.x_user_id == ""
        assert term.from_date_time == ""
        assert term.to_date_time == ""

    def test_partial_dict_fills_remaining_with_defaults(self):
        term = SearchTerm({"x_user_id": "123"})
        assert term.x_user_id == "123"
        assert term.from_date_time == ""
        assert term.to_date_time == ""


# ── to_dict ───────────────────────────────────────────────────────────────────

class TestSearchTermToDict:
    def test_returns_dict(self, search_term):
        assert isinstance(search_term.to_dict(), dict)

    def test_roundtrip(self, search_term, search_term_dict):
        assert search_term.to_dict() == search_term_dict

    def test_contains_all_keys(self, search_term):
        assert set(search_term.to_dict().keys()) == {
            "x_user_id", "from_date_time", "to_date_time"
        }


# ── to_json ───────────────────────────────────────────────────────────────────

class TestSearchTermToJson:
    def test_returns_valid_json_string(self, search_term):
        result = search_term.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_json_roundtrip(self, search_term, search_term_dict):
        assert json.loads(search_term.to_json()) == search_term_dict