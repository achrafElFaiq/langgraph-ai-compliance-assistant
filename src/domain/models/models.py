from datetime import date
from pydantic import BaseModel
from typing import Optional


class Article(BaseModel):
    regulation_name: str
    title_number: Optional[int]
    chapter_number: Optional[int]
    article_number: str
    article_title: Optional[str]
    breadcrumb: str
    content: str
    valid_from: date
    valid_until: Optional[date]
    source_url: str

class ArticleChunk(BaseModel):
    regulation_name: str
    title_number: Optional[int]
    chapter_number: Optional[int]
    article_number: str
    article_title: Optional[str]
    breadcrumb: str
    chunk_index: int
    chunk_total: int
    content: str
    valid_from: date
    valid_until: Optional[date]
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



class EvaluationResult(BaseModel):
    faithfulness: list[float]
    factual_correctness: list[float]
    context_recall: list[float]
    context_precision: list[float]
    end_to_end_latency: list[float] = []        # seconds per question
    node_latencies: dict[str, list[float]] = {} # node name -> seconds per execution
    retry_counts: list[int] = []                # retry_count per question
