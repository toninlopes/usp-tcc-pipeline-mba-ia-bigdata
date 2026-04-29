from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal
import urllib.request
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.database_tweets import DatabaseTweetsQuery
from base_model import BaseSentimentAnalyzer
from shared.text_cleaner import (
    remove_urls,
    replace_emojis_with_codes,
    replace_mentions,
    space_normalization,
    lowercase_normalization,
    remove_hashtags,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

SENTILEX_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "sentilex" / "sentiLex-PT02.txt"
)

# Tokens de negação — quando precedem uma palavra com polaridade, invertem o sinal.
# Lista baseada em Souza & Vieira (2012) adaptada ao domínio financeiro PT-BR.
NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "não",
        "nao",
        "nunca",
        "jamais",
        "nem",
        "nenhum",
        "nenhuma",
        "tampouco",
        "sequer",
    }
)

# Janela de negação: quantos tokens à frente da negação aplicar a inversão.
NEGATION_WINDOW: int = 3

URL = "https://raw.githubusercontent.com/sillasgonzaga/lexiconPT/refs/heads/master/data-raw/SentiLex-lem-PT02.txt"

BAR_WIDTH = 40

SentimentLabel = Literal["positivo", "negativo", "neutro"]

_POL_LABEL: dict[int, SentimentLabel] = {
    1: "positivo",
    -1: "negativo",
    0: "neutro",
}


# ---------------------------------------------------------------------------
# Parser do léxico
# ---------------------------------------------------------------------------

_POL_RE = re.compile(r"POL:N0=(-?\d+)")


# ---------------------------------------------------------------------------
# Lógica de pontuação
# ---------------------------------------------------------------------------


def _apply_negation(tokens: list[str], scores: list[int]) -> list[int]:
    """
    Inverte o sinal de tokens dentro da janela de negação.

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


def _score_to_label(score: int) -> SentimentLabel:
    if score > 0:
        return "positivo"
    if score < 0:
        return "negativo"
    return "neutro"


def _confidence(score: int, n_tokens: int) -> float:
    """
    Retorna um valor [0, 1] proporcional à magnitude do score normalizada
    pelo número de tokens. Serve apenas como proxy de confiança comparável
    ao campo 'score' retornado pelo FinBERT.
    """
    if n_tokens == 0:
        return 0.0
    return min(abs(score) / n_tokens, 1.0)


# ---------------------------------------------------------------------------
# Analisador principal
# ---------------------------------------------------------------------------


class SentiLexAnalyzer(BaseSentimentAnalyzer):
    """
    Classificador léxico de sentimento usando SentiLex-PT02.

    Estratégia de agregação:
        1. Pré-processa o texto com shared/text_cleaner.py.
        2. Tokeniza por espaços.
        3. Aplica janela de negação (NEGATION_WINDOW = 3 tokens).
        4. Soma as polaridades dos tokens encontrados no léxico.
           soma > 0  → positivo
           soma < 0  → negativo
           soma == 0 → neutro  (inclui textos sem cobertura léxica)
    """

    model_name = "SentiLex-PT02"
    classificator: str = "SentiLex-PT"

    def __init__(self, lexicon_path: Path = SENTILEX_PATH) -> None:
        super().__init__()
        self._lexicon_path = lexicon_path
        self.lexicon: dict[str, int] = {}
        self._db_tweets = DatabaseTweetsQuery(self.db_manager)
        self._model = self.load_model()

    def _download_lexicon_(self) -> None:
        """Baixa o léxico do SentiLex-PT02 usando o script oficial.

        O léxico é grande (~100k entradas) e não é recomendado para download
        programático frequente. Este método é pensado para uso manual, caso o
        arquivo seja perdido ou corrompido localmente.
        """
        print("Baixando SentiLex-PT02...")
        with urllib.request.urlopen(URL) as response:
            total = int(response.headers.get("Content-Length", 0))
            chunks: list[bytes] = []
            downloaded = 0
            chunk_size = 8 * 1024

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = downloaded / total
                    filled = int(BAR_WIDTH * pct)
                    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1%}  {downloaded / 1024:.1f} / {total / 1024:.1f} KB"
                    )
                else:
                    sys.stdout.write(f"\r  Baixando... {downloaded / 1024:.1f} KB")
                sys.stdout.flush()

            sys.stdout.write("\n")
            data = b"".join(chunks)
            print(f"Download concluído ({len(data) / 1024:.1f} KB).")

        SENTILEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        SENTILEX_PATH.write_bytes(data)
        print(f"Arquivo salvo em: {SENTILEX_PATH}")

        lines = SENTILEX_PATH.read_text(encoding="utf-8").splitlines()

        print("\nPrimeiras 3 linhas:")
        for line in lines[:3]:
            print(f"  {line}")

        print("\nSentiLex-PT02 pronto para uso.")
    # ── BaseSentimentAnalyzer interface ─────────────────────────────────────

    def load_model(self) -> dict[str, int]:
        """Carrega o léxico SentiLex-PT02 em memória."""

        if not SENTILEX_PATH.exists():
            self._download_lexicon_()
            sentilex_loaded = self.load_model()
            if not len(sentilex_loaded):
                raise FileNotFoundError(
                    f"SentiLex-PT02 não encontrado em '{SENTILEX_PATH}'.\n"
                    "Execute primeiro:  make download-sentilex"
                )
            return sentilex_loaded
        
        lexicon: dict[str, int] = {}

        with open(SENTILEX_PATH, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                try:
                    token_pos, rest = line.split(";", 1)
                    token = token_pos.rsplit(".", 1)[0].lower()
                    m = _POL_RE.search(rest)
                    print(f"Token: '{token}' | POL match on {rest}: {m}")
                    if m:
                        lexicon[token] = int(m.group(1))
                except (ValueError, IndexError):
                    continue
        
        print(f"Léxico carregado: {len(lexicon):,} tokens com polaridade.")

        return lexicon

    def preprocess(self, text: str) -> str:
        """
        Reutiliza shared/text_cleaner.py — mesma limpeza usada pelo FinBERT.
        Não aplica lematização aqui: o SentiLex-PT02 já contém formas flexionadas.
        """
        text = remove_urls(text)
        text = replace_emojis_with_codes(text)
        text = replace_mentions(text)
        text = remove_hashtags(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        return text

    def predict_text(
        self,
        text: str,
        batch_size: int = 1,  # ignorado — léxico é token-a-token
    ) -> list[dict]:
        """
        Classifica um único texto e retorna lista com um dict:
            [{"label": SentimentLabel, "score": float}]

        O campo 'score' é um proxy de confiança [0, 1], análogo ao retornado
        pelo FinBERT, para permitir comparações consistentes no dashboard.
        """
        if not self._model:
            raise RuntimeError("Léxico não carregado. Chame load_model() primeiro.")

        tokens = text.split() if text else []

        raw_scores = [self._model.get(token, 0) for token in tokens]
        adjusted_scores = _apply_negation(tokens, raw_scores)

        total = sum(adjusted_scores)
        label = _score_to_label(total)
        confidence = _confidence(total, len(tokens))

        return [{"label": label, "score": confidence}]

    def run(self) -> pd.DataFrame:
        """
        Pipeline completo:
            1. Carrega léxico.
            2. Busca tweets financeiros com anotação humana no banco.
            3. Classifica cada tweet com SentiLex-PT.
            4. Persiste resultados em tweets_classification
               (classificator = 'SentiLex-PT').
            5. Retorna DataFrame com os resultados.
        """
        rows = self._db_tweets.fetch_tweets_with_human_classification()

        if rows.empty:
            print("⚠  Nenhum tweet financeiro com anotação humana encontrado.")
            return pd.DataFrame()

        print(f"[SentiLex-PT02] Pré-processando {len(rows)} tweets...")

        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)

        print(f"⚙  Classificando {len(rows):,} tweets com SentiLex-PT...")

        # Compute predictions with a simple progress bar
        total = len(rows)
        predictions: list[list[dict]] = []
        for i, text in enumerate(rows["clear_tweets"], start=1):
            predictions.append(self.predict_text(text))

            # progress display
            pct = i / total
            filled = int(BAR_WIDTH * pct)
            bar_s = "█" * filled + "░" * (BAR_WIDTH - filled)
            sys.stdout.write(f"\r  [{bar_s}] {pct:5.1%}  {i}/{total}")
            sys.stdout.flush()

        sys.stdout.write("\n")
        rows["predicted_sentiment"] = predictions

        n_pos = (rows["predicted_sentiment"].str.contains("positivo")).sum()
        n_neg = (rows["predicted_sentiment"].str.contains("negativo")).sum()
        n_neu = (rows["predicted_sentiment"].str.contains("neutro")).sum()

        print(
            f"✔  Classificação concluída:\n"
            f"   ├─ Positivo : {n_pos:,}\n"
            f"   ├─ Negativo : {n_neg:,}\n"
            f"   └─ Neutro   : {n_neu:,}\n"
        )

        return rows


# ---------------------------------------------------------------------------
# Ponto de entrada CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyzer = SentiLexAnalyzer()
    result = analyzer.run()
    if not result.empty:
        print(result[["tweet_id", "clear_tweets", "predicted_sentiment"]].head(10))
