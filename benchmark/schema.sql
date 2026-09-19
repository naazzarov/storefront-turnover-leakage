-- Schema for the business_small table used by the Yelp text-to-SQL benchmark.
-- Mirrors the schema documented in app/core/openai_client.py.

CREATE TABLE IF NOT EXISTS business_small (
    business_id   VARCHAR PRIMARY KEY,
    name          VARCHAR,
    city          VARCHAR,
    state         VARCHAR,
    stars         FLOAT,
    review_count  INTEGER,
    categories    VARCHAR
);
