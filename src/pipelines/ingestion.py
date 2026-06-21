import time
import logging
import mlflow

from src.domain.ports.chunk import RegulationChunker
from src.domain.ports.embed import ArtcileEmbedder
from src.domain.ports.fetch import RegulationFetcher
from src.domain.ports.store import RegulationRepository
from src.config.init_regulations import REGULATIONS

logger = logging.getLogger(__name__)


async def run_ingestion_pipeline(
    fetcher: RegulationFetcher,
    chunker: RegulationChunker,
    embedder: ArtcileEmbedder,
    store: RegulationRepository
) -> None:
    start = time.perf_counter()
    logger.info("Ingestion pipeline started regulation_count=%s", len(REGULATIONS))

    mlflow.set_experiment("compliance-assistant-ingestion")

    with mlflow.start_run():
        mlflow.log_param("regulation_count", len(REGULATIONS))
        mlflow.log_param("regulations", ", ".join(r for r in REGULATIONS))
        mlflow.log_artifact("configs/regulations.yaml", artifact_path="config")

        await store.connect()
        await store.clear()

        try:
            total_articles = 0
            total_chunks = 0

            for step, regulation in enumerate(REGULATIONS, start=1):
                loop_start = time.perf_counter()
                logger.info("Regulation ingestion started alias=%s", regulation)

                result = await fetcher.fetch(regulation)
                if not result:
                    logger.warning("Empty result for regulation=%s, skipping", regulation)
                    continue

                article_ids = await store.store_articles(result.articles)
                chunks = await chunker.chunk(result.articles)
                chunks = await embedder.embed(chunks)
                await store.store_chunks(chunks, article_ids)

                duration_ms = int((time.perf_counter() - loop_start) * 1000)

                mlflow.log_metric("article_count", len(result.articles), step=step)
                mlflow.log_metric("chunk_count",   len(chunks),          step=step)
                mlflow.log_metric("duration_ms",   duration_ms,          step=step)
                mlflow.set_tag(f"regulation_{step}", regulation)

                total_articles += len(result.articles)
                total_chunks   += len(chunks)

                logger.info(
                    "Regulation ingestion completed alias=%s chunk_count=%s duration_ms=%s",
                    regulation, len(chunks), duration_ms,
                )

            total_ms = int((time.perf_counter() - start) * 1000)

            mlflow.log_metric("total_articles", total_articles)
            mlflow.log_metric("total_chunks",   total_chunks)
            mlflow.log_metric("total_duration_ms", total_ms)

            logger.info("Ingestion pipeline completed duration_ms=%s", total_ms)

        except Exception:
            mlflow.set_tag("status", "failed")
            logger.exception("Ingestion pipeline failed")
            raise
        finally:
            await store.close()