from typing import List

from src.domain.models.models import Article, ArticleChunk
from src.domain.ports.chunk import RegulationChunker

from langchain_text_splitters import RecursiveCharacterTextSplitter

class ArticleChunker(RegulationChunker):

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def chunk(self, articles: list[Article]) -> list[ArticleChunk]:


        result: list[ArticleChunk]  = []
        for article in articles:
            texts = self._splitter.split_text(article.content)
            chunk_total = len(texts)

            for i, text in enumerate(texts):
                result.append(ArticleChunk(
                    regulation_name=article.regulation_name,
                    title_number=article.title_number,
                    chapter_number=article.chapter_number,
                    article_number=article.article_number,
                    article_title=article.article_title,
                    breadcrumb=article.breadcrumb,
                    chunk_index=i,
                    chunk_total=chunk_total,
                    content=text,
                    valid_from=article.valid_from,
                    valid_until=article.valid_until,
                    source_url=article.source_url,
                ))

        return result

