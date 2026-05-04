from pydantic import BaseModel
from typing import Optional


class Article(BaseModel):
    regulation_name: str
    title_number: Optional[int]
    chapter_number: Optional[int]
    article_number: int
    article_title: Optional[str]
    breadcrumb: str
    content: str
    valid_from: str
    valid_until: Optional[str]
    source_url: str

class ArticleChunk(BaseModel):
    regulation_name: str
    title_number: Optional[int]
    chapter_number: Optional[int]
    article_number: int
    article_title: Optional[str]
    breadcrumb: str
    chunk_index: int
    chunk_total: int
    content: str
    valid_from: str
    valid_until: Optional[str]
    source_url: str
    embedding: Optional[list[float]] = None

class FetchResult(BaseModel):
    articles: list[Article]
    regulation_name: str
    valid_from: str
    source_url: str


class StoreResult(BaseModel):
    regulation_name: str
    article_count: int