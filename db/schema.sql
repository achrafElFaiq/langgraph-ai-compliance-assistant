CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    regulation_name TEXT NOT NULL,
    title_number    INTEGER,
    chapter_number  INTEGER,
    article_number  INTEGER NOT NULL,
    article_title   TEXT,
    breadcrumb      TEXT NOT NULL,
    content         TEXT NOT NULL,
    valid_from      DATE NOT NULL,
    valid_until     DATE,
    source_url      TEXT NOT NULL,
    CONSTRAINT unique_article UNIQUE (regulation_name, article_number)
);

CREATE TABLE IF NOT EXISTS article_chunks (
    id              SERIAL PRIMARY KEY,
    article_id      INTEGER NOT NULL REFERENCES articles(id),
    regulation_name TEXT NOT NULL,
    title_number    INTEGER,
    chapter_number  INTEGER,
    article_number  INTEGER NOT NULL,
    article_title   TEXT,
    breadcrumb      TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_total     INTEGER NOT NULL,
    content         TEXT NOT NULL,
    valid_from      DATE NOT NULL,
    valid_until     DATE,
    source_url      TEXT NOT NULL,
    embedding       vector(1024),
    CONSTRAINT unique_chunk UNIQUE (regulation_name, article_number, chunk_index)
);