from pydantic import BaseModel


class ChatRequest(BaseModel):
    input_text: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str