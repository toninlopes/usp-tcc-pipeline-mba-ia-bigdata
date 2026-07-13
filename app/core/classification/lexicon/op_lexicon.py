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
    lemmatize,
)

# ── Configuração ──────────────────────────────────────────────────────────────

# app/core/classification/ → app/core/ → app/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

LEXICON_PATH = _PROJECT_ROOT / "data" / "lexicons" / "oplexicon_v3.0" / "lexico_v3.0.txt"

OPLEXICON_URL = "https://raw.githubusercontent.com/marlovss/OpLexicon/refs/heads/main/lexico_v3.0.txt"

# ── Analisador ────────────────────────────────────────────────────────────────

class OpLexiconAnalyzer(LexiconSentimentAnalyzer):
    """Analisador de sentimento baseado no OpLexicon v3.0.

    Referência:
        SOUZA, M.; VIEIRA, R. Sentiment Analysis on Twitter with Portuguese
        Language Corpora. STIL, 2011.
    """

    model_name = "OpLexicon v3.0"
    classificator = "OpLexicon"
    _max_ngram_size = 3


    def __init__(self, lexicon_path: Path = LEXICON_PATH, lexicon_url: str = OPLEXICON_URL) -> None:
        super().__init__(lexicon_path=lexicon_path, lexicon_url=lexicon_url)


    def _load_model(self) -> Dict[str, int]:
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
        ''''
        Melhor configuração salva para OpLexicon — concordância 25.7%:
        Remover URLs, Remover emojis, Remover menções (@), Remover hashtags (#),
        Remover pontuação de borda (.!?,), Normalizar espaços,
        Normalizar caixa (preserva tickers),
        Remover stopwords, Lematizar
        '''
        text = remove_urls(text)
        text = remove_emojis(text)
        text = remove_mentions(text)
        text = remove_hashtags(text)
        text = strip_boundary_punctuation(text)
        text = space_normalization(text)
        text = lowercase_normalization(text)
        text = remove_stopwords(text)
        text = lemmatize(text)
        return text


if __name__ == "__main__":
    analyzer = OpLexiconAnalyzer()

    tweet = "Não gostei do resultado, mas o atendimento foi bom."
    # tweet = "🚀 O Itaú não apenas fechou bem 2025, como já traçou a rota para 2026. As novas projeções (guidance) mostram confiança no..."
    tweet = analyzer.preprocess(tweet)
    print(f"Texto pré-processado: {tweet}")

    label, score, matched_terms = analyzer.predict(tweet)
    print(f"Sentimento: {label}, Intensidade: {score:.4f}, Termos Correspondentes: {matched_terms}")