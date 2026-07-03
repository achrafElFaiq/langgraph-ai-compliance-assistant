import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from src.config.settings import setup_logging
from src.config.init_langfuse import langfuse_handler
from src.application.agent.graph import compiled_graph
from src.infrastructure.fetch.eurlex_fetch import EurLexFetcher
from src.infrastructure.chunk.text_chunk import ArticleChunker
from src.infrastructure.store.postgres_store import PostgresRegulationRepository
from src.infrastructure.embed.openrouter_embedder import OpenRouterEmbedder
from src.domain.ports.chunk import RegulationChunker
from src.domain.ports.embed import ArticleEmbedder
from src.domain.ports.fetch import RegulationFetcher
from src.domain.ports.store import RegulationRepository


async def run_ingestion():
    from src.pipelines.ingestion import run_ingestion_pipeline

    setup_logging()
    fetcher: RegulationFetcher = EurLexFetcher()
    store: RegulationRepository = PostgresRegulationRepository(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/compliance_db")
    )
    embedder: ArticleEmbedder = OpenRouterEmbedder("openai/text-embedding-3-small")
    chunker: RegulationChunker = ArticleChunker()
    await run_ingestion_pipeline(fetcher=fetcher, embedder=embedder, store=store, chunker=chunker)


async def run_query(question: str, thread_id: str = "test-1"):
    setup_logging()
    from src.config.init_store import store

    await store.connect()
    try:
        result = await compiled_graph.ainvoke(
            {"input_text": question},
            config={
                "callbacks": [langfuse_handler],
                "configurable": {"thread_id": thread_id}
            }
        )
    finally:
        await store.close()

    print(result.get("answer", result))


async def run_eval(dataset_path: str = "datasets/agent-eval/dataset.json"):
    from src.pipelines.evaluation import run_evaluation_pipeline

    setup_logging()
    await run_evaluation_pipeline(dataset_path=dataset_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utilities to test indexing, querying, and evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Run the ingestion/indexing pipeline.")

    query_parser = subparsers.add_parser("query", help="Run a single agent query.")
    query_parser.add_argument(
        "question",
        nargs="?",
        default="We are launching a crypto token in France, do we need a license?",
        help="Question to send to the compliance agent.",
    )
    query_parser.add_argument(
        "--thread-id",
        default="test-1",
        help="LangGraph thread identifier for the query run.",
    )

    eval_parser = subparsers.add_parser("eval", help="Run the evaluation pipeline.")
    eval_parser.add_argument(
        "--dataset",
        default="datasets/agent-eval/dataset.json",
        help="Path to the evaluation dataset JSON file.",
    )

    return parser


async def async_main() -> None:
    args = build_parser().parse_args()

    if args.command == "index":
        await run_ingestion()
    elif args.command == "query":
        await run_query(question=args.question, thread_id=args.thread_id)
    elif args.command == "eval":
        await run_eval(dataset_path=args.dataset)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
