from abc import abstractmethod
from typing import Any, Tuple

from app.core.processing.base_analyzer import BaseSentimentAnalyzer


class BertSentimentAnalyzer(BaseSentimentAnalyzer):
    """Base para modelos baseados em BERT via HuggingFace.

    Encapsula o carregamento do pipeline e a chamada de inferência,
    deixando para as subclasses apenas a definição de `model_name`,
    `load_model` e `preprocess`.

    Para adicionar um novo modelo BERT:

        class MyBertAnalyzer(BertSentimentAnalyzer):
            model_name = "org/my-bert-model"
            classificator = "MyBert"

            def load_model(self): ...
            def preprocess(self, text: str) -> str: ...
            def run(self) -> pd.DataFrame: ...
    """

    model_name: str

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None

    @abstractmethod
    def load_model(self) -> Any:
        """Carrega e retorna o pipeline HuggingFace."""
        ...

    def predict(self, text: str) -> Tuple[str, float]:
        """Executa inferência via pipeline HuggingFace.

        Trunca o texto em 512 tokens — limite do BERT.

        Returns:
            Tupla (label_pt, score).
        """
        result = self._model(text[:512], batch_size=1)[0]
        return self.normalize_label(result["label"]), result["score"]