from abc import abstractmethod, ABC

from src.domain.models.models import Article, ArticleChunk


class RegulationChunker(ABC):

    @abstractmethod
    async def chunk(self, articles: list[Article]) -> list[ArticleChunk]:
        pass