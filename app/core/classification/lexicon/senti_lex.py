from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from app.core.classification.lexicon.lexicon_analyzer import LexiconSentimentAnalyzer
from app.shared.text_cleaner import (
    remove_hashtags,
    remove_stopwords,
    remove_urls,
    remove_emojis,
    remove_mentions,
    strip_boundary_punctuation,
    space_normalization,
    lowercase_normalization,
)

# ── Configuração ──────────────────────────────────────────────────────────────

# app/core/classification/ → app/core/ → app/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# SENTILEX_PATH = _PROJECT_ROOT / "data" / "sentilex" / "sentiLex-PT02.txt"
SENTILEX_PATH = _PROJECT_ROOT / "data" / "sentilex" / "sentiLex-flex-PT02.txt"

# SENTILEX_URL = "https://raw.githubusercontent.com/sillasgonzaga/lexiconPT/refs/heads/master/data-raw/SentiLex-lem-PT02.txt"
SENTILEX_URL = "https://raw.githubusercontent.com/sillasgonzaga/lexiconPT/refs/heads/master/data-raw/SentiLex-flex-PT02.txt"

_POL_RE = re.compile(r"POL:N0=(-?\d+)")

class SentiLexAnalyzer(LexiconSentimentAnalyzer):
    """Classificador léxico de sentimento usando SentiLex-PT02.

    Referência:
        SANTOS, A. et al. SentiLex-PT: Principais características e potencialidades.
        Oslo Studies in Language, 2011.
    """

    model_name = "SentiLex-PT02"
    classificator = "SentiLex-PT"


    def __init__(self, lexicon_path: Path = SENTILEX_PATH) -> None:
        super().__init__(lexicon_path=lexicon_path, lexicon_url=SENTILEX_URL)


    def _load_model(self) -> Dict[str, int]:
        """Loads the SentiLex-PT02 lexicon from disk and returns a dict {lemma: polarity}."""
        if not self._lexicon_path.exists():
            self._download_lexicon()

        if not self._lexicon_path.exists():
            raise FileNotFoundError(
                f"SentiLex-PT02 não encontrado em '{self._lexicon_path}'.\n"
                "Execute: make download-sentilex"
            )

        lexicon: Dict[str, int] = {}
        with open(self._lexicon_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                try:
                    token_pos, rest = line.split(";", 1)
                    inflected = token_pos.split(",")[0].rsplit(".", 1)[0].lower()
                    m = _POL_RE.search(rest)
                    if m:
                        lexicon[inflected] = int(m.group(1))
                except (ValueError, IndexError):
                    continue

        print(f"[SentiLex] Léxico carregado: {len(lexicon):,} tokens.")
        return lexicon
    
    def preprocess(self, text: str) -> str:
        ''''
        Melhor configuração salva para SentiLex-PT — concordância 29.5%:
        Remover URLs, Remover emojis, Remover menções (@), Remover hashtags (#),
        Remover pontuação de borda (.!?,), Normalizar espaços,
        Normalizar caixa (preserva tickers), Remover stopwords
        '''
        text = remove_urls(text)
        text = remove_emojis(text)
        text = remove_mentions(text)
        text = remove_hashtags(text)
        text = strip_boundary_punctuation(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        text = remove_stopwords(text)
        return text


if __name__ == "__main__":
    analyzer = SentiLexAnalyzer()

    # print(list(analyzer._model.items())[:100])

    # print("gostei" in analyzer._model)

    # tweet = "Não gostei do resultado, mas o atendimento foi bom."
    tweet = "🚀 O Itaú não apenas fechou bem 2025, como já traçou a rota para 2026. As novas projeções (guidance) mostram confiança."
    tweet = analyzer.preprocess(tweet)
    print(f"Texto pré-processado: {tweet}")

    label, score, matched_terms = analyzer.predict(tweet)
    print(f"Sentimento: {label}, Intensidade: {score:.4f}, Termos Correspondentes: {matched_terms}")