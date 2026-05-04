import logging
from sentence_transformers import SentenceTransformer

from src.domain.models.models import Article, ArticleChunk
from src.domain.ports.embed import ArtcileEmbedder

logger = logging.getLogger(__name__)


class TransformersEmbedder(ArtcileEmbedder):

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    async def embed(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        if not chunks:
            return chunks

        texts = [chunk.content for chunk in chunks]
        logger.info("Embedding %d chunks...", len(texts))

        vectors = self._model.encode(texts, batch_size=4, show_progress_bar=True)

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector.tolist()

        logger.info("Embedding complete.")
        return chunks
