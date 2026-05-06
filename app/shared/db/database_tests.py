from unittest.mock import MagicMock, patch, call
import pytest

from app.shared.db.database import DatabaseManager


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def manager():
    """DatabaseManager com variáveis de ambiente padrão."""
    with patch("app.shared.db.database.load_dotenv"):
        return DatabaseManager()


@pytest.fixture
def mock_pool():
    """Pool de conexões simulado."""
    pool = MagicMock()
    pool.getconn.return_value = MagicMock()
    return pool


# ── _connection_params ────────────────────────────────────────────────────────

class TestConnectionParams:
    def test_returns_all_required_keys(self, manager):
        params = manager._connection_params()
        assert set(params.keys()) == {"host", "port", "dbname", "user", "password"}

    def test_reads_env_defaults(self, manager):
        params = manager._connection_params()
        assert params["dbname"] == "twitter_db"
        assert params["host"] == "localhost"
        assert params["port"] == 5432

    def test_reads_custom_env_vars(self):
        env = {
            "POSTGRES_DB": "custom_db",
            "POSTGRES_TWITTER_USER": "custom_user",
            "POSTGRES_TWITTER_PASSWORD": "secret",
            "POSTGRES_HOST": "db-server",
            "POSTGRES_PORT": "5433",
        }
        with patch("app.shared.db.database.load_dotenv"), \
             patch.dict("os.environ", env, clear=True):
            m = DatabaseManager()
            params = m._connection_params()

        assert params["dbname"] == "custom_db"
        assert params["user"] == "custom_user"
        assert params["password"] == "secret"
        assert params["host"] == "db-server"
        assert params["port"] == 5433

    def test_falls_back_to_postgres_user_when_twitter_user_absent(self):
        env = {"POSTGRES_USER": "fallback_user"}
        with patch("app.shared.db.database.load_dotenv"), \
             patch.dict("os.environ", env, clear=True):
            m = DatabaseManager()

        assert m._connection_params()["user"] == "fallback_user"


# ── initialize_pool ───────────────────────────────────────────────────────────

class TestInitializePool:
    def test_creates_pool_on_first_call(self, manager):
        with patch("app.shared.db.database.SimpleConnectionPool") as MockPool:
            manager.initialize_pool()
            MockPool.assert_called_once()

    def test_does_not_recreate_pool_on_second_call(self, manager, mock_pool):
        with patch("app.shared.db.database.SimpleConnectionPool") as MockPool:
            manager.initialize_pool()
            manager.initialize_pool()
            MockPool.assert_called_once()

    def test_passes_minconn_maxconn(self, manager):
        with patch("app.shared.db.database.SimpleConnectionPool") as MockPool:
            manager.initialize_pool(minconn=2, maxconn=5)
            args = MockPool.call_args
            assert args[0][0] == 2
            assert args[0][1] == 5


# ── check_connection ──────────────────────────────────────────────────────────

class TestCheckConnection:
    def test_returns_zero_on_success(self, manager):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        with patch("app.shared.db.database.psycopg2.connect", return_value=mock_conn):
            assert manager.check_connection() == 0

    def test_returns_one_on_failure(self, manager):
        with patch(
            "app.shared.db.database.psycopg2.connect",
            side_effect=Exception("connection refused"),
        ):
            assert manager.check_connection() == 1


# ── get_connection ────────────────────────────────────────────────────────────

class TestGetConnection:
    def test_yields_connection_from_pool(self, manager, mock_pool):
        manager._pool = mock_pool
        mock_conn = mock_pool.getconn.return_value

        with manager.get_connection() as conn:
            assert conn is mock_conn

    def test_returns_connection_to_pool_after_use(self, manager, mock_pool):
        manager._pool = mock_pool
        mock_conn = mock_pool.getconn.return_value

        with manager.get_connection() as conn:
            pass

        mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_returns_connection_to_pool_on_exception(self, manager, mock_pool):
        manager._pool = mock_pool
        mock_conn = mock_pool.getconn.return_value

        with pytest.raises(ValueError):
            with manager.get_connection():
                raise ValueError("erro simulado")

        mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_initializes_pool_if_not_ready(self, manager):
        assert manager._pool is None
        with patch.object(manager, "initialize_pool") as mock_init, \
             patch.object(manager, "_pool", create=True):
            manager._pool = MagicMock()
            manager._pool.getconn.return_value = MagicMock()
            with manager.get_connection():
                pass


# ── close_pool ────────────────────────────────────────────────────────────────

class TestClosePool:
    def test_closes_pool_when_exists(self, manager, mock_pool):
        manager._pool = mock_pool
        manager.close_pool()
        mock_pool.closeall.assert_called_once()

    def test_does_nothing_when_pool_is_none(self, manager):
        assert manager._pool is None
        manager.close_pool()  # não deve lançar exceção