import logging
from typing import List

import psycopg

from src.ingestion.models import Article

logger = logging.getLogger(__name__)


class EurLexStore:
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

    async def store(self, articles: List[Article]) -> None:
        if not self.connection:
            raise RuntimeError("Database connection is not initialized")

        if not articles:
            logger.info("No articles to store")
            return

        logger.info("Storing articles count=%s regulation=%s", len(articles), articles[0].regulation_name)
        try:
            for article in articles:
                await self._store_article(article)
            await self.connection.commit()
        except Exception:
            logger.exception("Failed while storing articles regulation=%s", articles[0].regulation_name)
            raise

        logger.info("Store completed count=%s regulation=%s", len(articles), articles[0].regulation_name)

    async def _store_article(self, article: Article) -> None:
        await self.connection.execute(
            """
            INSERT INTO regulations (regulation_name, title_number, chapter_number,
                                     article_number, article_title, breadcrumb,
                                     content, valid_from, valid_until, source_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (regulation_name, article_number) DO NOTHING
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
