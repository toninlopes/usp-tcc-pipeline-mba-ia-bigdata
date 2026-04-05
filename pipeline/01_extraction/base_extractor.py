from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """Interface abstrata para extratores de dados.
    Permite adicionar novas fontes (web scraping, RSS, etc.)
    sem modificar o código existente."""

    @abstractmethod
    def fetch(self, user_id: str, from_dt: str, to_dt: str) -> list[dict]:
        """Retorna lista de posts como dicionários padronizados."""
        ...
