import os
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class DatabaseManager:
    """Gerencia o pool de conexões com o PostgreSQL.

    Responsabilidade única: abrir, fornecer e fechar conexões.
    Queries ficam nos módulos db_*.py, que herdam desta classe.
    """

    def __init__(self):
        self._pool = None
        self._db_name = os.getenv("POSTGRES_DB", "twitter_db")
        self._db_user = os.getenv(
            "POSTGRES_TWITTER_USER", os.getenv("POSTGRES_USER", "postgres")
        )
        self._db_password = os.getenv(
            "POSTGRES_TWITTER_PASSWORD", os.getenv("POSTGRES_PASSWORD", "")
        )
        self._db_host = os.getenv("POSTGRES_HOST", "localhost")
        self._db_port = int(os.getenv("POSTGRES_PORT", "5432"))

    def _connection_params(self) -> dict:
        return {
            "host": self._db_host,
            "port": self._db_port,
            "dbname": self._db_name,
            "user": self._db_user,
            "password": self._db_password,
        }

    def initialize_pool(self, minconn: int = 1, maxconn: int = 10) -> None:
        """Inicializa o pool de conexões (chamado automaticamente se necessário)."""
        if self._pool is None:
            logger.info("Inicializando pool de conexões...")
            self._pool = SimpleConnectionPool(minconn, maxconn, **self._connection_params())
            logger.info("Pool de conexões inicializado.")

    def check_connection(self) -> int:
        """Verifica se a conexão com o banco está disponível.

        Returns:
            0 se bem-sucedido, 1 se falhou.
        """
        try:
            with psycopg2.connect(**self._connection_params()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            logger.info("Conexão com o banco verificada com sucesso.")
            return 0
        except Exception as exc:
            logger.error(f"Falha na conexão com o banco: {exc}")
            return 1

    @contextmanager
    def get_connection(self):
        """Context manager que fornece uma conexão do pool.

        Uso:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(...)
        """
        if self._pool is None:
            self.initialize_pool()

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn:
                self._pool.putconn(conn)

    def close_pool(self) -> None:
        """Fecha todas as conexões do pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Pool de conexões encerrado.")