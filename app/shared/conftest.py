from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import pytest


def make_repo(cls):
    """Instancia qualquer repositório sem conectar ao banco."""
    with patch("app.shared.database.load_dotenv"):
        return cls()


def mock_get_connection(repo, cursor: MagicMock):
    """Injeta um get_connection que fornece um cursor e conn simulados."""
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def fake_get_connection():
        yield mock_conn

    repo.get_connection = fake_get_connection
    return mock_conn


@pytest.fixture
def cursor():
    return MagicMock()