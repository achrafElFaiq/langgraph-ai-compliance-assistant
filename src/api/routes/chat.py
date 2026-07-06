from uuid import uuid4

from fastapi import APIRouter

from src.api.schemas.chat import ChatRequest, ChatResponse
from src.application.agent.graph import compiled_graph

router = APIRouter()

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a compliance query",
    description="Runs the full RAG pipeline synchronously and returns the complete answer. Use `/chat/stream` for streaming responses.",
    tags=["chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex

    result = await compiled_graph.ainvoke(
        {"input_text": request.input_text},
        config={"configurable": {"thread_id": thread_id}}
    )

    answer = result.get("answer", "") if isinstance(result, dict) else getattr(result, "answer", "")

    return ChatResponse(answer=answer, thread_id=thread_id)