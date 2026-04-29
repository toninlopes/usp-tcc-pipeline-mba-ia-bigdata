import sys
from pathlib import Path
import urllib.request
from numpy.ma import filled
import pandas as pd
from plotly.express import bar

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.database_tweets import DatabaseTweetsQuery
from shared.text_cleaner import (
    replace_urls,
    replace_emojis_with_codes,
    replace_mentions,
    space_normalization,
    lowercase_normalization,
)
from base_model import BaseSentimentAnalyzer

LEXICON_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "lexicons"
    / "oplexicon_v3.0"
    / "lexico_v3.0.txt"
)

URL = "https://raw.githubusercontent.com/marlovss/OpLexicon/refs/heads/main/lexico_v3.0.txt"

EXPECTED_LINES = 32_191

# Limiar de decisão: média de polaridade abaixo deste valor absoluto → neutro.
# Tweets financeiros tendem a ter muitos tokens sem entrada no léxico
# (siglas, tickers, nomes de emissores), o que puxa a média para zero.
# Um limiar baixo evita classificar como neutro textos com polaridade real
# diluída pelo vocabulário fora do léxico.
_THRESHOLD = 0.05

BAR_WIDTH = 40


class OpLexiconAnalyzer(BaseSentimentAnalyzer):
    """Analisador de sentimento baseado no OpLexicon v3.0.

    Estratégia de pontuação:
        1. Pré-processar o texto com o fluxo completo de clean() —
           lematização incluída, para maximizar a taxa de correspondência.
        2. Para cada token, buscar a polaridade no léxico.
        3. Agregar apenas os tokens com polaridade não-nula (média aritmética).
        4. Classificar conforme o sinal e magnitude da média.

    Referência:
        SOUZA, M.; VIEIRA, R. Sentiment Analysis on Twitter with Portuguese
        Language Corpora. STIL, 2011.
    """

    model_name = "OpLexicon v3.0"
    classificator = "OpLexicon"

    def __init__(self) -> None:
        super().__init__()
        self._db_tweets = DatabaseTweetsQuery(self.db_manager)
        self._model = self.load_model()

    def _download_lexicon_(self) -> None:
        """Baixa o léxico do OpLexicon v3.0 usando o script oficial.

        O léxico é grande (~100k entradas) e não é recomendado para download
        programático frequente. Este método é pensado para uso manual, caso o
        arquivo seja perdido ou corrompido localmente.
        """
        print("Baixando OpLexicon v3.0...")
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

        LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEXICON_PATH.write_bytes(data)
        print(f"Arquivo salvo em: {LEXICON_PATH}")

        lines = LEXICON_PATH.read_text(encoding="utf-8").splitlines()
        print(f"Linhas no léxico: {len(lines)} (esperado: ~{EXPECTED_LINES:,})")

        print("\nPrimeiras 3 linhas:")
        for line in lines[:3]:
            print(f"  {line}")

        print("\nOpLexicon v3.0 pronto para uso.")

    # ------------------------------------------------------------------
    # Interface BaseSentimentAnalyzer
    # ------------------------------------------------------------------

    def load_model(self) -> dict[str, int]:
        """Carrega o léxico do disco e retorna um dict {lema: polaridade}."""
        if not LEXICON_PATH.exists():
            self._download_lexicon_()
            lexicon_loaded = self.load_model()
            if not len(lexicon_loaded):
                raise FileNotFoundError(
                    f"Léxico não encontrado em {LEXICON_PATH}.\n"
                    "Execute primeiro: python data/lexicons/oplexicon_v3.0/download.py"
                )
            return lexicon_loaded

        lexicon: dict[str, int] = {}
        with LEXICON_PATH.open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue
                term, _, polarity_str = parts[0], parts[1], parts[2]
                try:
                    lexicon[term] = int(polarity_str)
                except ValueError:
                    continue

        print(f"[OpLexicon] Léxico carregado: {len(lexicon):,} entradas.")
        return lexicon

    def preprocess(self, text: str) -> str:
        """Aplica limpeza textual sem lematização (usado para texto individual)."""
        text = replace_urls(text)
        text = replace_emojis_with_codes(text)
        text = replace_mentions(text)
        # text = remove_hashtags(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        return text

    def predict_text(self, text: str, batch_size: int = 1) -> list[dict]:
        """Classifica um texto usando a pontuação léxica agregada.

        Retorna uma lista com um único dict {"label": str, "score": float},
        no mesmo formato que o FinBERTAnalyzer, para compatibilidade com
        BaseSentimentAnalyzer.save().

        Labels retornados: "POSITIVE" | "NEGATIVE" | "NEUTRAL"
        (normalize_label() da base converte para português).
        """
        tokens = text.split()
        scores = [self._model.get(token, 0) for token in tokens]
        relevant = [s for s in scores if s != 0]

        if not relevant:
            return [{"label": "NEUTRAL", "score": 0.0}]

        mean_score = sum(relevant) / len(relevant)

        if mean_score > _THRESHOLD:
            label = "POSITIVE"
            score = round(min(mean_score, 1.0), 4)
        elif mean_score < -_THRESHOLD:
            label = "NEGATIVE"
            score = round(min(abs(mean_score), 1.0), 4)
        else:
            label = "NEUTRAL"
            score = round(abs(mean_score), 4)

        return [{"label": label, "score": score}]

    def run(self) -> pd.DataFrame:
        """Busca tweets pendentes, executa a classificação e retorna as linhas.

        Seleciona apenas tweets que já possuem anotação humana e ainda não
        foram processados pelo OpLexicon, garantindo comparação direta com
        o gold standard.
        """
        rows = self._db_tweets.fetch_tweets_with_human_classification()

        if rows.empty:
            print("[OpLexicon] Nenhum tweet pendente encontrado.")
            return pd.DataFrame()

        print(f"[OpLexicon] Pré-processando {len(rows)} tweets...")

        rows["clear_tweets"] = rows["note_tweet"].apply(self.preprocess)

        print(f"[OpLexicon] Classificando {len(rows)} tweets...")

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

        return rows


if __name__ == "__main__":
    analyzer = OpLexiconAnalyzer()
    results = analyzer.run()
    if not results.empty:
        print(results[["tweet_id", "clear_tweets", "predicted_sentiment"]].head(10))
