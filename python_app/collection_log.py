from dataclasses import dataclass
import os
import json
from database import DatabaseManager

@dataclass
class SearchTerm:
    """Class to represent the search terms for collecting tweets."""

    x_user_id: str
    from_date_time: str
    to_date_time: str

    def __init__(self, json: dict):
        self.x_user_id = json.get('x_user_id', '')
        self.from_date_time = json.get('from_date_time', '')
        self.to_date_time = json.get('to_date_time', '')

    def to_dict(self):
        """Converts the SearchTerms to a dictionary format."""
        return {
            'x_user_id': self.x_user_id,
            'from_date_time': self.from_date_time,
            'to_date_time': self.to_date_time
        }
    
    def to_json(self):
        """Converts the SearchTerms to a JSON-serializable format."""
        return json.dumps(self.to_dict())

if __name__ == "__main__":

    try:
        # Caminho do script atual
        script_dir = os.path.dirname(__file__)  # pasta python_app/parse_tweet/
        print(f"Diretório do script: {script_dir}")

        # Subir 2 níveis: python_app/parse_tweet/ → python_app/ → projeto/
        project_root = os.path.dirname(script_dir)
        print(f"Raiz do projeto: {project_root}")

        # Caminho completo para o JSON
        json_path = os.path.join(project_root, "json_data", "search_tems.json")
        print(f"Caminho do JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)
    except FileNotFoundError:
        print(f"File not found: {json_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    search_terms: list[SearchTerm] = [SearchTerm(terms) for terms in json_data]

    db_manager = DatabaseManager()
    result = db_manager.check_connection()

    if result == 1:
        exit("Database connection failed. Exiting.")

    for term in search_terms:
        result = db_manager.query_log(term.to_dict())
        if len(result) == 0:
            record = db_manager.insert_log({
                'search_term': term.to_dict(),
                'tweets_collected': 0,
                'start_time': None,
                'end_time': None,
                'status': "pending",
                'error_message': None
            })

            if record:
                print(f"Log inserted for search term: {term.to_dict()}")
            else:
                print(f"Failed to insert log for search term: {term.to_dict()}")
        else:
            print(f"Log already exists for search term: {term.to_dict()}")
        