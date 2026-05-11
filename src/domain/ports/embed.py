from abc import ABC, abstractmethod
from src.domain.models.models import Article, ArticleChunk


class ArtcileEmbedder(ABC):


    @abstractmethod
    async def embed(self, articles: list[ArticleChunk]) -> list[ArticleChunk]:
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        pass