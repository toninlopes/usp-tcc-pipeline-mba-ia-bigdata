"""Helpers compartilhados entre os testes de repositório.

Separados do conftest.py para permitir importação direta nos módulos de teste
sem depender do mecanismo de descoberta automática do pytest.

O fixture `cursor` NÃO está aqui — ele é definido em conftest.py e
descoberto automaticamente pelo pytest para todos os arquivos do diretório.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch


def make_repo(cls):
    """Instancia qualquer repositório sem conectar ao banco."""
    with patch("app.shared.db.database.load_dotenv"):
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