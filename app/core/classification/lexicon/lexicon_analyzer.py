import sys
import urllib.request
from abc import abstractmethod
from typing import Dict, List, Tuple

from numpy.compat import Path
import pandas as pd

from app.core.classification.base_analyzer import BaseSentimentAnalyzer


# Tokens de negação — invertem a polaridade dentro da janela definida.
# Lista baseada em Souza & Vieira (2012) adaptada ao domínio financeiro PT-BR.
NEGATION_TOKENS: frozenset = frozenset({
    # --- Standard & Absolute Negations ---
    "não", "nem", "tampouco", "nunca", "jamais", "nada", "nenhum", "nenhuma", 
    "ninguém", "sequer", "absolutamente", "negativo",

    # --- Financial Prepositions & Restrictions (Without / Under) ---
    "sem", "contra", "menos", "exceto", "salvo", "fora", "rejeitado", "rejeitada",

    # --- Contractions & Variations (Common in conversational text/tweets) ---
    "n", "ñ", "nem-", "nehum", "nao", # Common typos/abbreviations

    # --- Multi-word Financial Negations (Phrasal Operators) ---
    "de forma alguma", "em hipótese alguma", "de maneira nenhuma", 
    "longe de", "passou longe de", "deixou de", "parou de", "pararam de",
    "ficou aquém de", "ficou abaixo de", "distante de", "passou batido por",
    "não conseguiu", "não conseguiram", "fracassou em", "falhou em"
})

# Janela de negação: quantos tokens à frente da negação aplicar a inversão.
NEGATION_WINDOW: int = 3

# Limiar de decisão: média de polaridade abaixo deste valor absoluto → neutro.
_THRESHOLD: float = 0.05

BAR_WIDTH = 40

class LexiconSentimentAnalyzer(BaseSentimentAnalyzer):
    """Base para analisadores de sentimento baseados em léxicos.

    Extrai o pipeline comum de run() — busca, pré-processamento,
    classificação com barra de progresso — deixando para as subclasses
    apenas a lógica de carregamento e pontuação do léxico específico.

    Para adicionar um novo léxico:

        class MyLexiconAnalyzer(LexiconSentimentAnalyzer):
            classificator = "MyLexicon"

            def _load_model(self) -> Dict[str, int]: ...
            def preprocess(self, text: str) -> str: ...
    """

    _model: Dict[str, int]
    _max_ngram_size = 3

    def __init__(self, lexicon_path: Path, lexicon_url: str) -> None:
        self._lexicon_path = lexicon_path
        self._lexicon_url = lexicon_url
        self._model: Dict[str, int] = self._load_model()
        self._max_ngram_size = self._extract_longest_ngrams_from_lexicon()

    @abstractmethod
    def _load_model(self) -> Dict[str, int]:
        """Load the lexicon from disk and return a dict {lemma: polarity}.

        Raises:
            FileNotFoundError: If the lexicon file is not found.

        Returns:
            Dict[str, int]: A dictionary mapping words/terms to their sentiment polarity scores.
        """
        ...

    def _download_lexicon(self):
        """Download the lexicon file from the specified URL and save it to disk.
        """

        print(f"Downloading {self.classificator} lexicon...")
        with urllib.request.urlopen(self._lexicon_url) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunks: list = []
            downloaded = 0
            while True:
                chunk = response.read(8 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total
                    filled = int(BAR_WIDTH * pct)
                    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
                    sys.stdout.write(f"\r  [{bar}] {pct:5.1%}  {downloaded/1024:.1f}/{total/1024:.1f} KB")
                sys.stdout.flush()
            sys.stdout.write("\n")
            data = b"".join(chunks)

        self._lexicon_path.parent.mkdir(parents=True, exist_ok=True)
        self._lexicon_path.write_bytes(data)
        print(f"File saved on: {self._lexicon_path}")

        
    def _extract_longest_ngrams_from_lexicon(self) -> int:
        """Extract the longest n-gram from the lexicon.

        Returns:
            int: The maximum n-gram size found in the lexicon.
        """
        if not self._model:
            return 1

        max_size = max(len(key.split()) for key in self._model)
        print(f"[{self.classificator}] Maior n-grama no léxico: {max_size} tokens.")
        return max_size
    

    def _extract_1_to_n_grams(self, words: List[str]) -> Tuple[int, List[str], List[Tuple[int, int]]]:
        """Calculates the score of a text by summing the polarities of the n-grams found in the lexicon.

        Args:
            words (List[str]): List of preprocessed tokens from the input text.

        Returns:
            Tuple[int, List[str], List[Tuple[int, int]]]: A tuple containing the total score, a list of matched terms with their scores, and a list of match positions with their scores.
        """
        score = 0
        matched_terms = []
        match_positions: List[Tuple[int, int]] = []  # (start_idx, span_score)
        skip_idx = set()

        for n in range(self._max_ngram_size, 0, -1):
            for i in range(len(words) - n + 1):
                if any(idx in skip_idx for idx in range(i, i + n)):
                    continue

                ngram = ' '.join(words[i:i+n])
                span_score = self._model.get(ngram)
                if span_score is not None:
                    score += span_score
                    matched_terms.append(f"'{ngram}'({span_score})")
                    match_positions.append((i, span_score))
                    for idx in range(i, i + n):
                        skip_idx.add(idx)

        return score, matched_terms, match_positions
    
    def _apply_negation(self, tokens: List[str], scores: List[int]) -> List[int]:
        """Inverte o sinal de tokens dentro da janela de negação.

        Exemplo:
            tokens = ["não", "bom", "resultado"]
            scores = [  0,    +1,       0      ]
            → scores corrigidos = [0, -1, 0]
        """
        adjusted = scores[:]
        negate_until = -1
        for i, tok in enumerate(tokens):
            if tok in NEGATION_TOKENS:
                negate_until = i + NEGATION_WINDOW
            elif i <= negate_until and adjusted[i] != 0:
                adjusted[i] = -adjusted[i]
        return adjusted


    def _confidence(self, score: int, n_tokens: int) -> float:
        """Proxy de confiança [0, 1] proporcional à magnitude do score."""
        if n_tokens == 0:
            return 0.0
        return min(abs(score) / n_tokens, 1.0)

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Pre-process the input text by applying a series of transformations to prepare it for sentiment analysis.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        ...

    def predict(self, text: str) -> Tuple[str, float, List[str]]:
        """Classify the sentiment of a text based on the intensity of polarity of its tokens in the lexicon.

        Returns:
            Tuple (label_pt, score) where label_pt is 'positivo', 'negativo' or
            'neutro' and score is the intensity of polarity in [0, 1].
        """
        tokens = text.split() if text else []
        if not tokens:
            return "neutro", 0.0, []

        _, matched_terms, match_positions = self._extract_1_to_n_grams(tokens)
        token_scores = [0] * len(tokens)
        for start_idx, span_score in match_positions:
            token_scores[start_idx] = span_score

        if not any(s != 0 for s in token_scores):
            return "neutro", 0.0, matched_terms

        adjusted = self._apply_negation(tokens, token_scores)

        terms_out: List[str] = []
        for (start_idx, original_score), term in zip(match_positions, matched_terms):
            adj_score = adjusted[start_idx]
            if adj_score != original_score:
                term_name = term[:term.rfind("(")]
                terms_out.append(f"{term_name}({adj_score})*neg")
            else:
                terms_out.append(term)

        relevant = [s for s in adjusted if s != 0]
        mean_score = sum(relevant) / len(relevant)

        if mean_score > _THRESHOLD:
            return "positivo", round(min(mean_score, 1.0), 4), terms_out
        elif mean_score < -_THRESHOLD:
            return "negativo", round(min(abs(mean_score), 1.0), 4), terms_out
        else:
            return "neutro", round(abs(mean_score), 4), terms_out


    def run(self) -> pd.DataFrame:
        """Pipeline comum: busca tweets, classifica e retorna resultados.

        Seleciona tweets financeiros com anotação humana para permitir
        comparação direta com o gold standard na avaliação.
        """
        rows = self._tweet_repo.query_all_tweets_with_human_classification()

        if rows.empty:
            print(f"[{self.classificator}] Nenhum tweet encontrado.")
            return pd.DataFrame()

        rows = rows[rows["has_human_classification"] == True].reset_index(drop=True)

        if rows.empty:
            print(f"[{self.classificator}] Nenhum tweet com anotação humana.")
            return pd.DataFrame()

        print(f"[{self.classificator}] Pré-processando {len(rows)} tweets...")
        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)

        print(f"[{self.classificator}] Classificando {len(rows)} tweets...")
        total = len(rows)
        predictions = []

        for i, text in enumerate(rows["clear_tweets"], start=1):
            label, score, matched_terms = self.predict(text)
            predictions.append([{"label": label, "score": score, "matched_terms": matched_terms}])
            pct = i / total
            filled = int(BAR_WIDTH * pct)
            bar = "█" * filled + "░" * (BAR_WIDTH - filled)
            sys.stdout.write(f"\r  [{bar}] {pct:5.1%}  {i}/{total}")
            sys.stdout.flush()

        sys.stdout.write("\n")
        rows["predicted_sentiment"] = predictions
        return rows