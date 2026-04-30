import os

import requests
from dotenv import load_dotenv

from app.core.extraction.base_extractor import BaseExtractor

load_dotenv()


class TwitterAPIExtractor(BaseExtractor):
    """Extrator de tweets via API X v2. Implementa BaseExtractor."""

    _bearer_token: str
    _headers: dict
    _params: dict
    _base_url: str

    def __init__(
        self,
        x_user_id: str,
        from_date_time: str,
        to_date_time: str,
        next_token: str = "",
    ):
        """
        Instancia o extrator para um usuário e período específicos.

        :param x_user_id: ID do usuário no X.
        :param from_date_time: Timestamp UTC mais antigo. Formato ISO 8601/RFC 3339.
        :param to_date_time: Timestamp UTC mais recente. Formato ISO 8601/RFC 3339.
        :param next_token: Token de paginação para a próxima página de resultados.
        """
        self._bearer_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self._base_url = f"https://api.x.com/2/users/{x_user_id}/tweets"
        self._headers = {"Authorization": f"Bearer {self._bearer_token}"}
        self._params = {
            "tweet.fields": "created_at,note_tweet,author_id,public_metrics,lang,source,entities,context_annotations,geo",
            "max_results": 100,
            "user.fields": "name,username,location,description,public_metrics",
            "start_time": from_date_time,
            "end_time": to_date_time,
        }

        if next_token:
            self._params["pagination_token"] = next_token

    def fetch(self, user_id: str, from_dt: str, to_dt: str) -> list[dict]:
        """Implementa BaseExtractor.fetch. Retorna lista de tweets como dicionários."""
        extractor = TwitterAPIExtractor(user_id, from_dt, to_dt)
        return [extractor.make_request()]

    def make_request(self) -> dict:
        """Executa a requisição à API X v2 e retorna a resposta como dicionário."""
        response = requests.get(
            url=self._base_url,
            headers=self._headers,
            params=self._params,
        )
        if response.status_code != 200:
            raise Exception(
                f"Request returned an error: {response.status_code} {response.text}"
            )
        return response.json()