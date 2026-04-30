import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from app.core.processing.lexicon_analyzer import LexiconSentimentAnalyzer
from app.shared.text_cleaner import (
    replace_urls,
    replace_emojis_with_codes,
    replace_mentions,
    space_normalization,
    lowercase_normalization,
)

# ── Configuração ──────────────────────────────────────────────────────────────

# app/core/processing/ → app/core/ → app/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

LEXICON_PATH = _PROJECT_ROOT / "data" / "lexicons" / "oplexicon_v3.0" / "lexico_v3.0.txt"

OPLEXICON_URL = "https://raw.githubusercontent.com/marlovss/OpLexicon/refs/heads/main/lexico_v3.0.txt"

EXPECTED_LINES = 32_191

BAR_WIDTH = 40

# Limiar de decisão: média de polaridade abaixo deste valor absoluto → neutro.
# Tweets financeiros têm muitos tokens fora do léxico (siglas, tickers),
# o que dilui a média. Um limiar baixo evita classificar como neutro textos
# com polaridade real diluída pelo vocabulário fora do léxico.
_THRESHOLD = 0.05


# ── Analisador ────────────────────────────────────────────────────────────────

class OpLexiconAnalyzer(LexiconSentimentAnalyzer):
    """Analisador de sentimento baseado no OpLexicon v3.0.

    Estratégia de pontuação:
        1. Pré-processa o texto com shared/text_cleaner.py.
        2. Para cada token, busca a polaridade no léxico.
        3. Agrega apenas tokens com polaridade não-nula (média aritmética).
        4. Classifica conforme sinal e magnitude da média vs. _THRESHOLD.

    Referência:
        SOUZA, M.; VIEIRA, R. Sentiment Analysis on Twitter with Portuguese
        Language Corpora. STIL, 2011.
    """

    model_name = "OpLexicon v3.0"
    classificator = "OpLexicon"

    def __init__(self, lexicon_path: Path = LEXICON_PATH) -> None:
        self._lexicon_path = lexicon_path
        super().__init__()
        self._model: Dict[str, int] = self.load_model()

    def _download_lexicon(self) -> None:
        """Baixa o OpLexicon v3.0 do repositório oficial."""
        print("Baixando OpLexicon v3.0...")
        with urllib.request.urlopen(OPLEXICON_URL) as response:
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
        print(f"Arquivo salvo em: {self._lexicon_path}")
        lines = self._lexicon_path.read_text(encoding="utf-8").splitlines()
        print(f"Linhas: {len(lines):,} (esperado: ~{EXPECTED_LINES:,})")

    def load_model(self) -> Dict[str, int]:
        """Carrega o léxico do disco e retorna dict {lema: polaridade}."""
        if not self._lexicon_path.exists():
            self._download_lexicon()

        if not self._lexicon_path.exists():
            raise FileNotFoundError(
                f"Léxico não encontrado em {self._lexicon_path}.\n"
                "Execute: make download-oplexicon"
            )

        lexicon: Dict[str, int] = {}
        with self._lexicon_path.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue
                term, polarity_str = parts[0], parts[2]
                try:
                    lexicon[term] = int(polarity_str)
                except ValueError:
                    continue

        print(f"[OpLexicon] Léxico carregado: {len(lexicon):,} entradas.")
        return lexicon

    def preprocess(self, text: str) -> str:
        """Limpeza textual para OpLexicon.

        Mantém hashtags como texto simples (remove apenas o símbolo #) para
        ampliar a cobertura léxica com termos financeiros frequentemente hashtagueados.
        """
        text = replace_urls(text)
        text = replace_emojis_with_codes(text)
        text = replace_mentions(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        return text

    def predict(self, text: str) -> Tuple[str, float]:
        """Classifica um texto pela média de polaridade dos tokens no léxico.

        Tokens ausentes no léxico são ignorados na agregação, evitando que
        vocabulário fora do domínio dilua a polaridade dos tokens relevantes.

        Returns:
            Tupla (label_pt, score) onde label_pt é 'positivo', 'negativo' ou 'neutro'
            e score é a magnitude da média, capturada em [0, 1].
        """
        tokens = text.split() if text else []
        scores = [self._model.get(token, 0) for token in tokens]
        relevant = [s for s in scores if s != 0]

        if not relevant:
            return "neutro", 0.0

        mean_score = sum(relevant) / len(relevant)

        if mean_score > _THRESHOLD:
            return "positivo", round(min(mean_score, 1.0), 4)
        elif mean_score < -_THRESHOLD:
            return "negativo", round(min(abs(mean_score), 1.0), 4)
        else:
            return "neutro", round(abs(mean_score), 4)


if __name__ == "__main__":
    analyzer = OpLexiconAnalyzer()
    results = analyzer.run()
    if not results.empty:
        print(results[["tweet_id", "clear_tweets", "predicted_sentiment"]].head(10))