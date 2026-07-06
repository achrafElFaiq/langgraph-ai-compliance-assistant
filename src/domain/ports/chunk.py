from abc import ABC, abstractmethod

from src.domain.models.models import Article, ArticleChunk


class RegulationChunker(ABC):

    @abstractmethod
    def chunk(self, articles: list[Article]) -> list[ArticleChunk]:
        pass