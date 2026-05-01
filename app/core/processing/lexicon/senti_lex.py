from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from app.core.processing.lexicon.lexicon_analyzer import LexiconSentimentAnalyzer
from app.shared.text_cleaner import (
    replace_urls,
    replace_emojis_with_codes,
    replace_mentions,
    remove_hashtags,
    space_normalization,
    lowercase_normalization,
)

# ── Configuração ──────────────────────────────────────────────────────────────

# app/core/processing/ → app/core/ → app/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

SENTILEX_PATH = _PROJECT_ROOT / "data" / "sentilex" / "sentiLex-PT02.txt"

SENTILEX_URL = "https://raw.githubusercontent.com/sillasgonzaga/lexiconPT/refs/heads/master/data-raw/SentiLex-lem-PT02.txt"

# Tokens de negação — invertem a polaridade dentro da janela definida.
# Lista baseada em Souza & Vieira (2012) adaptada ao domínio financeiro PT-BR.
NEGATION_TOKENS: frozenset = frozenset({
    "não", "nao", "nunca", "jamais", "nem",
    "nenhum", "nenhuma", "tampouco", "sequer",
})

# Janela de negação: quantos tokens à frente da negação aplicar a inversão.
NEGATION_WINDOW: int = 3

BAR_WIDTH = 40

_POL_RE = re.compile(r"POL:N0=(-?\d+)")


# ── Funções auxiliares ────────────────────────────────────────────────────────

def _apply_negation(tokens: List[str], scores: List[int]) -> List[int]:
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


def _confidence(score: int, n_tokens: int) -> float:
    """Proxy de confiança [0, 1] proporcional à magnitude do score."""
    if n_tokens == 0:
        return 0.0
    return min(abs(score) / n_tokens, 1.0)


# ── Analisador ────────────────────────────────────────────────────────────────

class SentiLexAnalyzer(LexiconSentimentAnalyzer):
    """Classificador léxico de sentimento usando SentiLex-PT02.

    Estratégia de agregação:
        1. Pré-processa o texto com shared/text_cleaner.py.
        2. Tokeniza por espaços.
        3. Aplica janela de negação (NEGATION_WINDOW = 3 tokens).
        4. Soma as polaridades dos tokens encontrados no léxico:
               soma > 0  → positivo
               soma < 0  → negativo
               soma == 0 → neutro  (inclui textos sem cobertura léxica)

    Referência:
        SANTOS, A. et al. SentiLex-PT: Principais características e potencialidades.
        Oslo Studies in Language, 2011.
    """

    model_name = "SentiLex-PT02"
    classificator = "SentiLex-PT"

    def __init__(self, lexicon_path: Path = SENTILEX_PATH) -> None:
        self._lexicon_path = lexicon_path
        super().__init__()
        self._model: Dict[str, int] = self.load_model()

    def _download_lexicon(self) -> None:
        """Baixa o SentiLex-PT02 do repositório lexiconPT."""
        print("Baixando SentiLex-PT02...")
        with urllib.request.urlopen(SENTILEX_URL) as response:
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

    def load_model(self) -> Dict[str, int]:
        """Carrega o léxico SentiLex-PT02 em memória."""
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
                    token = token_pos.rsplit(".", 1)[0].lower()
                    m = _POL_RE.search(rest)
                    if m:
                        lexicon[token] = int(m.group(1))
                except (ValueError, IndexError):
                    continue

        print(f"[SentiLex] Léxico carregado: {len(lexicon):,} tokens.")
        return lexicon

    def preprocess(self, text: str) -> str:
        """Limpeza textual para SentiLex.

        Não aplica lematização pois o SentiLex-PT02 já contém formas flexionadas.
        """
        text = replace_urls(text)
        text = replace_emojis_with_codes(text)
        text = replace_mentions(text)
        text = remove_hashtags(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        return text

    def predict(self, text: str) -> Tuple[str, float]:
        """Classifica um texto via pontuação léxica com tratamento de negação.

        Returns:
            Tupla (label_pt, score) onde label_pt é 'positivo', 'negativo' ou 'neutro'
            e score é um proxy de confiança em [0, 1].
        """
        tokens = text.split() if text else []
        raw_scores = [self._model.get(token, 0) for token in tokens]
        adjusted = _apply_negation(tokens, raw_scores)
        total = sum(adjusted)

        if total > 0:
            label = "positivo"
        elif total < 0:
            label = "negativo"
        else:
            label = "neutro"

        return label, _confidence(total, len(tokens))


if __name__ == "__main__":
    analyzer = SentiLexAnalyzer()
    result = analyzer.run()
    if not result.empty:
        print(result[["tweet_id", "clear_tweets", "predicted_sentiment"]].head(10))