from abc import ABC, abstractmethod
from typing import List

from src.domain.models.models import Article, ArticleChunk


class RegulationRepository(ABC):

    @abstractmethod
    async def connect(self) -> None:
        pass


    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def store_articles(self, articles: List[Article]) -> dict[int, int]:
        pass

    @abstractmethod
    async def store_chunks(self, chunks: List[ArticleChunk], article_ids: dict[int, int]) -> None:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def retrieve(self, embedding: list[float], query: str, top_k: int = 10) -> list[Article]:
        pass
