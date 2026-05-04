import logging
from typing import List

import psycopg

from src.domain.models.models import Article, ArticleChunk
from src.domain.ports.store import RegulationRepository

logger = logging.getLogger(__name__)

class PostresRegulationRepository(RegulationRepository):

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None

    async def connect(self):
        logger.info("Opening database connection")
        try:
            self.connection = await psycopg.AsyncConnection.connect(self.connection_string)
        except Exception:
            logger.exception("Database connection failed")
            raise
        logger.info("Database connection ready")

    async def close(self):
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")

    async def clear(self) -> None:
        await self.connection.execute("DELETE FROM regulations")
        await self.connection.commit()

    async def store(self, chunks: List[ArticleChunk]) -> None:
        if not self.connection:
            raise RuntimeError("Database connection is not initialized")

        if not chunks:
            logger.info("No chunks to store")
            return

        logger.info("Storing chunks count=%s regulation=%s", len(chunks), chunks[0].regulation_name)
        try:
            for chunk in chunks:
                await self._store_article(chunk)
            await self.connection.commit()
        except Exception:
            logger.exception("Failed while storing chunks regulation=%s", chunks[0].regulation_name)
            raise

        logger.info("Store completed count=%s regulation=%s", len(chunks), chunks[0].regulation_name)

    async def _store_article(self, chunk: ArticleChunk) -> None:
        await self.connection.execute(
            """
            INSERT INTO regulations (regulation_name, title_number, chapter_number,
                                     article_number, article_title, breadcrumb,
                                     chunk_index, chunk_total, content,
                                     valid_from, valid_until, source_url, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s) ON CONFLICT (regulation_name, article_number, chunk_index) DO NOTHING
            """,
            (
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