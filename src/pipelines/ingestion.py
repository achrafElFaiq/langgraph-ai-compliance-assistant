import asyncio
import json
import os
import tempfile
import time
import logging
from pathlib import Path

import mlflow

from src.domain.ports.chunk import RegulationChunker
from src.domain.ports.embed import ArticleEmbedder
from src.domain.ports.fetch import RegulationFetcher
from src.domain.ports.store import RegulationRepository
from src.config.init_regulations import REGULATIONS

logger = logging.getLogger(__name__)


async def _ingest_one(
    regulation: str,
    fetcher: RegulationFetcher,
    chunker: RegulationChunker,
    embedder: ArticleEmbedder,
    store: RegulationRepository,
) -> dict:
    loop_start = time.perf_counter()
    logger.info("Regulation ingestion started alias=%s", regulation)

    result = await fetcher.fetch(regulation)
    if not result:
        logger.warning("Empty result for regulation=%s, skipping", regulation)
        return {"regulation": regulation, "articles": 0, "chunks": 0, "duration_ms": 0, "skipped": True}

    article_ids = await store.store_articles(result.articles)
    chunks = chunker.chunk(result.articles)
    chunks = await embedder.embed(chunks)
    await store.store_chunks(chunks, article_ids)

    duration_ms = int((time.perf_counter() - loop_start) * 1000)
    logger.info(
        "Regulation ingestion completed alias=%s articles=%d chunks=%d duration_ms=%d",
        regulation, len(result.articles), len(chunks), duration_ms,
    )
    return {"regulation": regulation, "articles": len(result.articles), "chunks": len(chunks), "duration_ms": duration_ms, "skipped": False}


async def run_ingestion_pipeline(
    fetcher: RegulationFetcher,
    chunker: RegulationChunker,
    embedder: ArticleEmbedder,
    store: RegulationRepository
) -> None:
    start = time.perf_counter()
    logger.info("Ingestion pipeline started regulation_count=%d", len(REGULATIONS))

    mlflow.set_experiment("compliance-assistant-ingestion")

    with mlflow.start_run():
        mlflow.log_param("regulation_count", len(REGULATIONS))
        mlflow.log_param("regulations", ", ".join(r for r in REGULATIONS))
        mlflow.log_artifact("configs/regulations.yaml", artifact_path="config")

        await store.connect()
        await store.clear()

        try:
            results = await asyncio.gather(*[
                _ingest_one(regulation, fetcher, chunker, embedder, store)
                for regulation in REGULATIONS
            ])

            total_articles = 0
            total_chunks = 0
            per_regulation: dict[str, dict] = {}

            for r in results:
                reg = r["regulation"]
                per_regulation[reg] = {
                    "articles": r["articles"],
                    "chunks": r["chunks"],
                    "duration_ms": r["duration_ms"],
                    "skipped": r["skipped"],
                }
                mlflow.set_tag(f"regulation_{reg}", "skipped" if r["skipped"] else "ingested")
                if r["skipped"]:
                    continue
                # One named metric per regulation → each shows on its own (not collapsed by step).
                mlflow.log_metric(f"articles_{reg}",    r["articles"])
                mlflow.log_metric(f"chunks_{reg}",      r["chunks"])
                mlflow.log_metric(f"duration_ms_{reg}", r["duration_ms"])
                total_articles += r["articles"]
                total_chunks   += r["chunks"]

            total_ms = int((time.perf_counter() - start) * 1000)

            mlflow.log_metric("total_articles",    total_articles)
            mlflow.log_metric("total_chunks",      total_chunks)
            mlflow.log_metric("total_duration_ms", total_ms)

            # Per-regulation breakdown as a tabular artifact for richer inspection.
            with tempfile.TemporaryDirectory() as tmp:
                breakdown_path = os.path.join(tmp, "per_regulation.json")
                Path(breakdown_path).write_text(
                    json.dumps(per_regulation, indent=2, ensure_ascii=False)
                )
                mlflow.log_artifact(breakdown_path, artifact_path="results")

            logger.info(
                "Ingestion pipeline completed total_articles=%d total_chunks=%d duration_ms=%d per_regulation=%s",
                total_articles, total_chunks, total_ms,
                {reg: v["articles"] for reg, v in per_regulation.items()},
            )

        except Exception:
            mlflow.set_tag("status", "failed")
            logger.exception("Ingestion pipeline failed")
            raise
        finally:
            await store.close()
