import pandas as pd
from psycopg2.extras import Json
from loguru import logger

from app.shared.db.database import DatabaseManager


class CollectionLogRepository(DatabaseManager):
    """Queries e operações sobre a tabela `collection_log`."""

    def query_log(self, search_term: dict) -> pd.DataFrame:
        """Retorna registros de log que correspondem ao search_term informado.

        Usa o operador JSONB @> para checar se o search_term é subconjunto
        do valor armazenado, permitindo buscas parciais por x_user_id ou período.

        Args:
            search_term: Dicionário com chaves x_user_id, from_date_time, to_date_time.
        """
        query = """
        SELECT * FROM collection_log
        WHERE search_term @> %s::jsonb
        ORDER BY start_time DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (Json(search_term) if search_term else None,))
                    results = cur.fetchall()
                    logger.info(f"Logs consultados para search_term '{search_term}'.")
                    return pd.DataFrame(results, columns=self._log_columns())
        except Exception as e:
            logger.error(f"Falha ao consultar logs para search_term '{search_term}': {e}")
            return pd.DataFrame()

    def query_all_pending_logs(self) -> pd.DataFrame:
        """Retorna todos os registros com status 'pending', ordenados pelo período mais recente.

        Usado pelo extrator para determinar quais coletas ainda precisam ser executadas.
        """
        query = """
        SELECT * FROM collection_log
        WHERE status = 'pending'
        ORDER BY (search_term->>'to_date_time') DESC;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    results = cur.fetchall()
                    logger.info("Logs pendentes consultados com sucesso.")
                    return pd.DataFrame(results, columns=self._log_columns())
        except Exception as e:
            logger.error(f"Falha ao consultar logs pendentes: {e}")
            return pd.DataFrame()

    def insert_log(self, log_data: dict) -> bool:
        """Insere um novo registro na tabela `collection_log`.

        Args:
            log_data: Dicionário com as chaves search_term, tweets_collected,
                      start_time, end_time, status, error_message.

        Returns:
            True se inserido com sucesso, False caso contrário.
        """
        query = """
        INSERT INTO collection_log
            (search_term, tweets_collected, start_time, end_time, status, error_message)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            Json(log_data["search_term"]) if log_data.get("search_term") else None,
                            log_data.get("tweets_collected"),
                            log_data.get("start_time"),
                            log_data.get("end_time"),
                            log_data.get("status"),
                            log_data.get("error_message"),
                        ),
                    )
                    conn.commit()
                    logger.info("Registro de log inserido com sucesso.")
                    return True
        except Exception as e:
            logger.error(f"Falha ao inserir registro de log: {e}")
            conn.rollback()
            return False

    def update_log(self, log_id: int, update_data: dict) -> bool:
        """Atualiza campos de um registro existente em `collection_log`.

        Usado pelo extrator para registrar start_time, end_time, tweets_collected
        e status ao final de cada coleta.

        Args:
            log_id: Chave primária do registro a atualizar.
            update_data: Dicionário com os campos e valores a atualizar.
                         Ex: {'status': 'completed', 'tweets_collected': 42}

        Returns:
            True se atualizado com sucesso, False caso contrário.
        """
        set_clause = ", ".join([f"{key} = %s" for key in update_data.keys()])
        query = f"UPDATE collection_log SET {set_clause} WHERE id = %s"

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, [*update_data.values(), log_id])
                    conn.commit()
                    logger.info(f"Log ID {log_id} atualizado com sucesso.")
                    return True
        except Exception as e:
            logger.error(f"Falha ao atualizar log ID {log_id}: {e}")
            conn.rollback()
            return False

    @staticmethod
    def _log_columns() -> list[str]:
        """Colunas da tabela collection_log na ordem retornada pelo banco."""
        return [
            "id",
            "search_term",
            "tweets_collected",
            "start_time",
            "end_time",
            "status",
            "error_message",
        ]