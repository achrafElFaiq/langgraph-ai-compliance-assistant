from pydantic import BaseModel


class ChatRequest(BaseModel):
    input_text: str
    thread_id: str | None = None


class Citation(BaseModel):
    breadcrumb: str
    relevant: bool
    excerpts: list[str]


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    regulations: list[str] = []
    route: str = ""
    retry_count: int = 0
    fallback_attempted: bool = False
    citations: list[Citation] = []
    final_report: str = ""