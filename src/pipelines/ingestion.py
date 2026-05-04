import time
import logging

from src.core.ingestion.character_chunker import ArticleChunker
from src.domain.ports.chunk import RegulationChunker
from src.domain.ports.embed import ArtcileEmbedder
from src.domain.ports.fetch import RegulationFetcher
from src.domain.ports.store import RegulationRepository
from src.config.config import REGULATIONS


logger = logging.getLogger(__name__)


async def run_ingestion_pipeline(fetcher: RegulationFetcher, chunker: RegulationChunker, embedder: ArtcileEmbedder, store: RegulationRepository) -> None:
    start = time.perf_counter()
    logger.info("Ingestion pipeline started regulation_count=%s", len(REGULATIONS))

    await store.connect()
    await store.clear()

    try:
        for regulation in REGULATIONS:
            loop_start = time.perf_counter()
            logger.info("Regulation ingestion started alias=%s", regulation)

            result = await fetcher.fetch(regulation)
            if not result:
                logger.warning("Empty result for regulation=%s, skipping", regulation)
                continue

            chunks = await chunker.chunk(result.articles)
            chunks = await embedder.embed(chunks)
            await store.store(chunks)

            duration_ms = int((time.perf_counter() - loop_start) * 1000)
            logger.info(
                "Regulation ingestion completed alias=%s chunk_count=%s duration_ms=%s",
                regulation,
                len(chunks),
                duration_ms,
            )

        total_ms = int((time.perf_counter() - start) * 1000)
        logger.info("Ingestion pipeline completed duration_ms=%s", total_ms)
    except Exception:
        logger.exception("Ingestion pipeline failed")
        raise
    finally:
        await store.close()


