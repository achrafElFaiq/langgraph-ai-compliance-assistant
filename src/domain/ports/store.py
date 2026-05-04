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
    async def store(self, articles: List[ArticleChunk]) -> None:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass