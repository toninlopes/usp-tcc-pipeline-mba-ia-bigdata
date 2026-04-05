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
    CONSTRAINT fk_tweet FOREIGN KEY (tweet_id) REFERENCES tweets(id) ON DELETE CASCADE
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

-- Confirmação
DO $$
BEGIN
    RAISE NOTICE 'Banco twitter_db configurado com sucesso para o usuário twitter_user';
END $$;
