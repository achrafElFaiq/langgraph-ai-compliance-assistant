import logging
from sentence_transformers import SentenceTransformer

from src.domain.models.models import ArticleChunk
from src.domain.ports.embed import ArtcileEmbedder

logger = logging.getLogger(__name__)


class TransformersEmbedder(ArtcileEmbedder):
    """Sentence-Transformers implementation of the embedding port.

    Loads a transformer model once and enriches `ArticleChunk` items with
    vector embeddings.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        """Initialize and load the embedding model by name."""
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    async def embed(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        """Generate embeddings for each chunk and return the same list updated."""
        if not chunks:
            return chunks

        texts = [chunk.content for chunk in chunks]
        logger.info("Embedding %d chunks...", len(texts))

        vectors = self._model.encode(texts, batch_size=4, show_progress_bar=True)

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector.tolist()

        logger.info("Embedding complete.")
        return chunks
