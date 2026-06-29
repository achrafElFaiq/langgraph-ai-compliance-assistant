import logging
import os

from openai import AsyncOpenAI

from src.domain.models.models import ArticleChunk
from src.domain.ports.embed import ArticleEmbedder

logger = logging.getLogger(__name__)


class OpenRouterEmbedder(ArticleEmbedder):
    """OPen router implementation of the embedding port."""

    def __init__(self, model_name: str = "openai/text-embedding-3-small") -> None:
        """Initialize OpenAI client for embeddings."""
        self.model_name = model_name

        # Support both OpenRouter and direct OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY required")

        # Use OpenRouter endpoint if using OpenRouter key
        if os.getenv("OPENROUTER_API_KEY"):
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            self._client = AsyncOpenAI(api_key=api_key)

        logger.info("OpenAI embeddings initialized: %s", model_name)

    async def embed(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        """Generate embeddings via API for each chunk."""
        if not chunks:
            return chunks

        texts = [chunk.content for chunk in chunks]
        logger.info("Embedding %d chunks via API...", len(texts))

        # Call OpenAI API (batched)
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        # Map embeddings back to chunks
        for chunk, item in zip(chunks, response.data):
            chunk.embedding = item.embedding

        logger.info("Embedding complete.")
        return chunks

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        response = await self._client.embeddings.create(
            model=self.model_name,
            input=[query]
        )
        return response.data[0].embedding
