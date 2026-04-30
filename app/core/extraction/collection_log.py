import json
import os
from dataclasses import dataclass
from pathlib import Path

from app.shared.db_collection_log import CollectionLogRepository


@dataclass
class SearchTerm:
    """Representa os parâmetros de busca para coleta de tweets."""

    x_user_id: str
    from_date_time: str
    to_date_time: str

    def __init__(self, data: dict):
        self.x_user_id = data.get("x_user_id", "")
        self.from_date_time = data.get("from_date_time", "")
        self.to_date_time = data.get("to_date_time", "")

    def to_dict(self) -> dict:
        return {
            "x_user_id": self.x_user_id,
            "from_date_time": self.from_date_time,
            "to_date_time": self.to_date_time,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


if __name__ == "__main__":
    # app/core/extraction/ → app/core/ → app/ → project root
    project_root = Path(__file__).resolve().parents[3]
    json_path = project_root / "config" / "search_terms_monthly.json"

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"Arquivo não encontrado: {json_path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Erro ao decodificar JSON: {e}")

    search_terms = [SearchTerm(term) for term in json_data]

    repo = CollectionLogRepository()
    if repo.check_connection() == 1:
        raise SystemExit("Falha na conexão com o banco. Encerrando.")

    for term in search_terms:
        existing = repo.query_log(term.to_dict())
        if existing.empty:
            inserted = repo.insert_log({
                "search_term": term.to_dict(),
                "tweets_collected": 0,
                "start_time": None,
                "end_time": None,
                "status": "pending",
                "error_message": None,
            })
            if inserted:
                print(f"Log inserido: {term.to_dict()}")
            else:
                print(f"Falha ao inserir log: {term.to_dict()}")
        else:
            print(f"Log já existe: {term.to_dict()}")