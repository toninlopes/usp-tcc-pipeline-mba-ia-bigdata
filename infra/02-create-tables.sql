CREATE TABLE IF NOT EXISTS tweets (
    id SERIAL PRIMARY KEY,
    tweet_id VARCHAR(50) UNIQUE NOT NULL,
    username VARCHAR(100) NOT NULL,
    note_tweet TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    likes INTEGER DEFAULT 0,
    hashtags JSONB,
    tweet JSONB,
    sentiment VARCHAR(8), -- e.g., 'positivo', 'negativo', 'neutro'
    is_finance_news INTEGER -- e.g., 1 para notícias financeiras, 0 para outras
);
CREATE INDEX idx_tweets_username ON tweets(username);
CREATE INDEX idx_tweets_created_at ON tweets(created_at);
COMMENT ON TABLE tweets IS 'Armazena tweets coletados do X/Twitter';

CREATE TABLE IF NOT EXISTS tweets_classification (
    id SERIAL PRIMARY KEY,
    tweet_id SERIAL NOT NULL,
    sentiment VARCHAR(8), -- e.g., 'positivo', 'negativo', 'neutro'
    why_sentiment TEXT, -- Justificativa para a classificação de sentimento
    is_finance_news INTEGER, -- e.g., 1 para notícias financeiras, 0 para outras
    why_is_finance_news TEXT, -- Justificativa para classificar como notícia financeira
    classificator VARCHAR(100), -- e.g., 'Sonnet 4.6', 'Gemini 3', 'ChatGPT 4.0', etc.
    score REAL, -- Score de confiança da classificação (se aplicável)
    CONSTRAINT fk_tweet FOREIGN KEY (tweet_id) REFERENCES tweets(id) ON DELETE CASCADE,
    CONSTRAINT uq_tweet_classificator UNIQUE (classificator)
);

CREATE INDEX idx_tweets_classification_id ON tweets_classification(tweet_id);
COMMENT ON TABLE tweets_classification IS 'Armazena classificações de tweets';

CREATE TABLE IF NOT EXISTS collection_log (
    id SERIAL PRIMARY KEY,
    search_term JSONB NOT NULL,
    tweets_collected INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20), -- e.g., 'pending', 'collected'
    error_message TEXT
);
COMMENT ON TABLE collection_log IS 'Log de controle das coletas realizadas';

CREATE TABLE IF NOT EXISTS dataset_split (
    id       SERIAL PRIMARY KEY,
    tweet_id INTEGER NOT NULL REFERENCES tweets(id) ON DELETE CASCADE,
    split    VARCHAR(10) NOT NULL CHECK (split IN ('train', 'test')),
    fold     SMALLINT    DEFAULT NULL,
    CONSTRAINT uq_dataset_split UNIQUE (tweet_id),
    CONSTRAINT chk_fold CHECK (
        (split = 'test'  AND fold IS NULL)          OR
        (split = 'train' AND fold BETWEEN 1 AND 4)
    )
);

CREATE INDEX IF NOT EXISTS idx_dataset_split_tweet_id ON dataset_split(tweet_id);
CREATE INDEX IF NOT EXISTS idx_dataset_split_split    ON dataset_split(split);
CREATE INDEX IF NOT EXISTS idx_dataset_split_fold     ON dataset_split(fold);

COMMENT ON TABLE dataset_split IS
    'Particionamento estratificado do dataset de tweets com anotação humana. '
    'split=test: hold-out fixo para avaliação comparativa entre modelos. '
    'split=train: conjunto de fine-tuning, subdividido em 4 folds para K-Fold estratificado.';

COMMENT ON COLUMN dataset_split.fold IS
    'Fold do K-Fold (1–4) para tweets de train. NULL para tweets do hold-out (test).';


-- Confirmação
DO $$
BEGIN
    RAISE NOTICE 'Banco twitter_db configurado com sucesso para o usuário twitter_user';
END $$;
