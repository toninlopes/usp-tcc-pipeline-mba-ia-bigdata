SELECT * FROM tweets t
WHERE EXISTS (
  SELECT 1 FROM jsonb_array_elements(t.hashtags->'hashtags') elem
  WHERE upper(elem->>'tag') IN (
    'ECONOMIA', 'ACIONISTAS', 'INVESTIMENTOS', 'BOLSA', 'ASSETS', 'ATIVOS', 'AÇÕES',
    'B3', 'BC', 'BACEN', 'BALANÇO', 'BALANÇOS', 'BANCOCENTRAL',  'BANCOS', 'BENS',
    'BITCOIN', 'BLOOMBERG', 'BOLSA', 'BOLSADEVALORES', 'BUSINESS', 'CADE', 'CLIMA',
    'COMMODITIES', 'COMPRAS', 'COMÉRCIO', 'COMBUSTÍVEL', 'COMBUSTÍVEIS', 'COPOM',
    'CRIPTO', 'CRIPTOS', 'CRIPTOATIVOS', 'CRIPTOMOEDA', 'CRIPTOMOEDAS',
    'CRISE', 'CRISEECONOMICA', 'CRÉDITO', 'DIVIDAS', 'DEBÊNTURES', 'DEFICITPRIMÁRIO',
    'DESEMPREGO', 'DEVEDOR', 'DINHEIRO', 'DIVIDENDOS', 'DOWJONES', 'DÉCIMOTERCEIRO',
    'DÉFICIT', 'DÉFICITPRIMÁRIO', 'DÍVIDA', 'DÍVIDAPÚBLICA', 'DÓLAR', 'EARNINGS',
    'EDUCAÇÃO', 'EDUCAÇÃOFINANCEIRA', 'EMPREENDEDORISMO', 'EMPREGO', 'EMPREGOS',
    'EMPRESA', 'EMPRESAS', 'EMPRÉSTIMO', 'EXPORTAÇÃO', 'EXPORTAÇÕES', 'FII', 'FIIS',
    'FMI', 'FALÊNCIA', 'FARIALIMA', 'FAZENDA', 'FEBRABAN', 'FECHAMENTO', 'FECHAMENTODEMERCADO',
    'FIAGRO', 'FIAGROS', 'FIIS', 'FINANCAS', 'FINANCEIRO', 'FINANCIAMENTO', 'FINANCIAMENTOIMOBILIÁRIO',
    'FINANÇA', 'FINANÇAS', 'FINTECH', 'FINTECHS', 'FITCH', 'FOCUS', 'FOMC', 'FORTUNA', 'FORTUNAS',
    'FRAUDE', 'FRAUDES', 'FUNDOS', 'FUNDOSIMOBILIÁRIOS', 'FUNDOSDEINVESTIMENTO', 'GASTO',
    'HABITAÇÃO', 'IA', 'IBGE', 'IFIX', 'IGP', 'IMC', 'IMPOSTO', 'INCA', 'INPC', 'INPI',
    'IOF', 'IPC', 'IPCA', 'IPO', 'IR', 'ISS', 'IBOV', 'IBOVESPA', 'IMPOSTO', 'IMPOSTODERENDA',
    'IMPOSTOS', 'IMÓVEIS', 'IMÓVEL', 'INADIMPLÊNCIA', 'INDÚSTRIA', 'INFLACAO', 'INFLAÇÃO',
    'INOVAÇÃO', 'INSTABILIDADE', 'INTELIGENCIAARTIFICIAL', 'INTELIGÊNCIA', 'INTELIGÊNCIAARTIFICIAL',
    'INTERNETBANK', 'INVESTIMENTO', 'INVESTIMENTOS', 'ISENÇÃO'
    )
);

SELECT DISTINCT elem->>'tag' AS tag FROM tweets t,
jsonb_array_elements(hashtags->'hashtags') AS elem
ORDER BY tag;


SELECT l.* FROM collection_log l;

INSERT INTO collection_log (search_term) VALUES ('{"x_user_id": "51150679", "to_date_time": "2026-02-12T03:00:00Z", "from_date_time": "2025-09-01T02:59:00Z"}');

DELETE FROM collection_log WHERE tweets_collected IS NULL;

SELECT l.search_term->>'x_user_id' FROM collection_log l;


-- SELECT tablename, tableowner 
-- FROM pg_tables 
-- WHERE tablename = 'tweets';

-- SELECT current_user;

-- ALTER TABLE tweets ADD COLUMN sentiment VARCHAR(8); -- e.g., 'positivo', 'negativo', 'neutro'
-- ALTER TABLE tweets ADD COLUMN is_finance_tweet INTEGER; -- 1 para notícias financeiras, 0 para outras


SELECT MAX(t.created_at), MIN(t.created_at) FROM tweets t
ORDER BY created_at DESC;

SELECT
    t.*,
    EXISTS (
        SELECT 1 FROM tweets_classification tc
        WHERE tc.tweet_id = t.id
          AND tc.classificator = 'Humano'
    ) AS has_human_classification
FROM tweets t
ORDER by t.created_at DESC;


-- UPDATE tweets SET is_finance_tweet = NULL, sentiment = NULL

SELECT * FROM tweets t ORDER BY t.created_at DESC

SELECT * FROM tweets_classification c
ORDER BY tweet_id DESC;


WITH tweet_modes AS (
    SELECT 
        tweet_id,
        -- Finds the most frequent value for is_finance_news
        MODE() WITHIN GROUP (ORDER BY is_finance_news) AS dominant_finance,
        -- Finds the most frequent value for sentiment
        MODE() WITHIN GROUP (ORDER BY sentiment) AS dominant_sentiment
    FROM tweets_classification
    GROUP BY tweet_id
)
UPDATE tweets t
SET 
    is_finance_tweet = m.dominant_finance,
    sentiment = m.dominant_sentiment
FROM tweet_modes m
WHERE t.id = m.tweet_id
  AND t.is_finance_tweet IS NULL;


WITH classified AS (
    SELECT
        t.*,
        EXISTS (
            SELECT 1 FROM tweets_classification tc
            WHERE tc.tweet_id = t.id
            AND tc.classificator = 'Humano'
        ) AS has_human_classification,
        EXISTS (
            SELECT 1 FROM tweets_classification tc
            WHERE tc.tweet_id = t.id
            AND tc.classificator = 'FinBERT-PT-BR'
        ) AS has_finbert_classification
    FROM tweets t
)

SELECT t.* FROM classified t
WHERE is_finance_tweet = 1 AND has_human_classification = TRUE
AND has_finbert_classification = FALSE
ORDER BY t.created_at DESC;


SELECT * FROM tweets_classification
WHERE classificator = 'FinBERT-PT-BR';

DELETE FROM tweets_classification
WHERE classificator = 'FinBERT-PT-BR';


SELECT t.note_tweet FROM tweets t
WHERE note_tweet ~ '\m[A-Z]{4,5}[0-9]{1,2}\M';
