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



class FetchResult(BaseModel):
    html: str
    regulation_name: str
    valid_from: str
    source_url: str