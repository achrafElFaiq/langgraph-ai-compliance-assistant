import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from src.config.settings import setup_logging
from src.application.agent.graph import compiled_graph
from src.infrastructure.eurlex_fetch import EurLexFetcher
from src.infrastructure.text_chunk import ArticleChunker
from src.infrastructure.postgres_store import PostgresRegulationRepository
from src.infrastructure.transformers_embed import TransformersEmbedder
from src.domain.ports.chunk import RegulationChunker
from src.domain.ports.embed import ArtcileEmbedder
from src.domain.ports.fetch import RegulationFetcher
from src.domain.ports.store import RegulationRepository
from src.pipelines.ingestion import run_ingestion_pipeline


async def run_ingestion():
    setup_logging()
    fetcher: RegulationFetcher = EurLexFetcher()
    store: RegulationRepository = PostgresRegulationRepository(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/compliance_db")
    )
    embedder: ArtcileEmbedder = TransformersEmbedder(model_name="BAAI/bge-m3")
    chunker: RegulationChunker = ArticleChunker()
    await run_ingestion_pipeline(fetcher=fetcher, embedder=embedder, store=store, chunker=chunker)


if __name__ == "__main__":
    from src.config.init_store import store

    async def test():
        setup_logging()
        await store.connect()
        result = await compiled_graph.ainvoke(
            {"input_text": "We are launching a crypto token in France, do we need a license?"},
            config={"configurable": {"thread_id": "test-1"}}
        )
        print(result["answer"])

    asyncio.run(test())
