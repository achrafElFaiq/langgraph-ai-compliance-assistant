import asyncio
import logging
import os

from src.core.ingestion import setup_logging
from src.core.ingestion.character_chunker import ArticleChunker
from src.core.ingestion.eurlex_fetcher import EurLexFetcher
from src.core.ingestion.postgres_store import PostresRegulationRepository
from src.core.ingestion.transformers_embedder import TransformersEmbedder
from src.domain.ports.chunk import RegulationChunker
from src.domain.ports.embed import ArtcileEmbedder
from src.domain.ports.fetch import RegulationFetcher
from src.domain.ports.store import RegulationRepository
from src.pipelines.ingestion import run_ingestion_pipeline



async def main():
    setup_logging()
    fetcher: RegulationFetcher = EurLexFetcher()
    store: RegulationRepository = PostresRegulationRepository(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/compliance_db")
    )
    embedder: ArtcileEmbedder = TransformersEmbedder(model_name = "BAAI/bge-m3")
    chunker: RegulationChunker = ArticleChunker()
    await run_ingestion_pipeline(fetcher=fetcher,embedder= embedder, store=store, chunker=chunker)



if __name__ == "__main__":
    asyncio.run(main())