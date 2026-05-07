import logging
from typing import List

import psycopg

from src.domain.models.models import Article, ArticleChunk
from src.domain.ports.store import RegulationRepository

logger = logging.getLogger(__name__)

class PostgresRegulationRepository(RegulationRepository):
    """PostgreSQL-backed implementation of `RegulationRepository`.

    Persists articles and chunks, and provides hybrid retrieval combining
    vector similarity with full-text ranking.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None

    async def connect(self):
        """Open an async PostgreSQL connection for repository operations."""
        logger.info("Opening database connection")
        try:
            self.connection = await psycopg.AsyncConnection.connect(self.connection_string)
        except Exception:
            logger.exception("Database connection failed")
            raise
        logger.info("Database connection ready")

    async def close(self):
        """Close the active database connection if one is open."""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")

    async def clear(self) -> None:
        """Delete all stored chunks and articles from persistence tables."""
        await self.connection.execute("DELETE FROM article_chunks")
        await self.connection.execute("DELETE FROM articles")
        await self.connection.commit()

    async def store_articles(self, articles: List[Article]) -> dict[int, int]:
        """Store articles and return a map of article number to database id."""
        if not self.connection:
            raise RuntimeError("Database connection is not initialized")

        if not articles:
            return {}

        logger.info("Storing articles count=%s regulation=%s", len(articles), articles[0].regulation_name)
        article_ids = {}
        try:
            for article in articles:
                article_id = await self._insert_article(article)
                article_ids[article.article_number] = article_id
            await self.connection.commit()
        except Exception:
            logger.exception("Failed while storing articles regulation=%s", articles[0].regulation_name)
            raise

        logger.info("Articles stored count=%s", len(article_ids))
        return article_ids

    async def _insert_article(self, article: Article) -> int:
        async with self.connection.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO articles (regulation_name, title_number, chapter_number,
                                      article_number, article_title, breadcrumb,
                                      content, valid_from, valid_until, source_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (regulation_name, article_number) DO NOTHING
                RETURNING id
                """,
                (
                    article.regulation_name,
                    article.title_number,
                    article.chapter_number,
                    article.article_number,
                    article.article_title,
                    article.breadcrumb,
                    article.content,
                    article.valid_from,
                    article.valid_until,
                    article.source_url,
                )
            )
            row = await cur.fetchone()
            return row[0]

    async def store_chunks(self, chunks: List[ArticleChunk], article_ids: dict[int, int]) -> None:
        """Store article chunks using the provided article-number to id mapping."""
        if not self.connection:
            raise RuntimeError("Database connection is not initialized")

        if not chunks:
            logger.info("No chunks to store")
            return

        logger.info("Storing chunks count=%s regulation=%s", len(chunks), chunks[0].regulation_name)
        try:
            for chunk in chunks:
                article_id = article_ids[chunk.article_number]
                await self._insert_chunk(chunk, article_id)
            await self.connection.commit()
        except Exception:
            logger.exception("Failed while storing chunks regulation=%s", chunks[0].regulation_name)
            raise

        logger.info("Chunks stored count=%s", len(chunks))

    async def _insert_chunk(self, chunk: ArticleChunk, article_id: int) -> None:
        await self.connection.execute(
            """
            INSERT INTO article_chunks (article_id, regulation_name, title_number, chapter_number,
                                        article_number, article_title, breadcrumb,
                                        chunk_index, chunk_total, content,
                                        valid_from, valid_until, source_url, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s) ON CONFLICT (regulation_name, article_number, chunk_index) DO NOTHING
            """,
            (
                article_id,
                chunk.regulation_name,
                chunk.title_number,
                chunk.chapter_number,
                chunk.article_number,
                chunk.article_title,
                chunk.breadcrumb,
                chunk.chunk_index,
                chunk.chunk_total,
                chunk.content,
                chunk.valid_from,
                chunk.valid_until,
                chunk.source_url,
                chunk.embedding,
            )
        )

    async def retrieve(self, embedding: list[float], query: str, top_k: int = 10) -> list[Article]:
        """Retrieve top matching articles using hybrid vector + BM25 ranking."""
        if not self.connection:
            raise RuntimeError("Database connection is not initialized")

        async with self.connection.cursor() as cur:
            await cur.execute(
                """
                WITH vector_search AS (SELECT article_id,
                                              ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS rank
                                       FROM article_chunks
                                       ORDER BY embedding
                    <=> %s::vector
                    LIMIT %s
                    )
                   , bm25_search AS (
                SELECT article_id, ROW_NUMBER() OVER (ORDER BY ts_rank(to_tsvector('french', content), plainto_tsquery('french', %s)) DESC) AS rank
                FROM article_chunks
                WHERE to_tsvector('french', content) @@ plainto_tsquery('french', %s)
                ORDER BY ts_rank(to_tsvector('french', content), plainto_tsquery('french', %s)) DESC
                    LIMIT %s
                    ), rrf AS (
                SELECT
                    COALESCE (v.article_id, b.article_id) AS article_id, COALESCE (1.0 / (60 + v.rank), 0) + COALESCE (1.0 / (60 + b.rank), 0) AS rrf_score
                FROM vector_search v
                    FULL OUTER JOIN bm25_search b
                ON v.article_id = b.article_id
                    )
                SELECT a.regulation_name,
                       a.title_number,
                       a.chapter_number,
                       a.article_number,
                       a.article_title,
                       a.breadcrumb,
                       a.content,
                       a.valid_from,
                       a.valid_until,
                       a.source_url
                FROM rrf
                         JOIN articles a ON a.id = rrf.article_id
                ORDER BY rrf_score DESC
                    LIMIT %s
                """,
                (embedding, embedding, top_k, query, query, query, top_k, top_k)
            )

            rows = await cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return [Article(**dict(zip(columns, row))) for row in rows]
