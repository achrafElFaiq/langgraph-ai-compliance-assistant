import asyncio
import logging
import os
import time

from src.ingestion import setup_logging
from src.ingestion.config import REGULATIONS
from src.ingestion.fetch import EurLexFetcher
from src.ingestion.parse import EurLexParser
from src.ingestion.store import EurLexStore

logger = logging.getLogger(__name__)


async def main():
    setup_logging()

    fetcher = EurLexFetcher()
    parser = EurLexParser()
    connection_string = os.getenv("DATABASE_URL", "postgresql://localhost/compliance_db")
    store = EurLexStore(connection_string=connection_string)

    start = time.perf_counter()
    logger.info("Ingestion pipeline started regulation_count=%s", len(REGULATIONS))

    await store.connect()
    try:
        for regulation in REGULATIONS:
            loop_start = time.perf_counter()
            logger.info("Regulation ingestion started alias=%s", regulation)

            result = await fetcher.fetch(regulation)
            articles = parser.parse(result)
            await store.store(articles)

            duration_ms = int((time.perf_counter() - loop_start) * 1000)
            logger.info(
                "Regulation ingestion completed alias=%s regulation=%s article_count=%s duration_ms=%s",
                regulation,
                result.regulation_name,
                len(articles),
                duration_ms,
            )
    except Exception:
        logger.exception("Ingestion pipeline failed")
        raise
    finally:
        await store.close()

    total_ms = int((time.perf_counter() - start) * 1000)
    logger.info("Ingestion pipeline completed duration_ms=%s", total_ms)


if __name__ == "__main__":
    asyncio.run(main())