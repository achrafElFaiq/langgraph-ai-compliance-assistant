from fastapi import APIRouter
from src.config.init_store import store
from src.config.init_llm import llm

router = APIRouter()
@router.get(
    "/health",
    summary="Liveness check",
    description="Returns 200 if the API is running.",
    tags=["health"],
)
async def health():
    return {"status": "ok"}


@router.get(
    "/health/ready",
    summary="Readiness check",
    description="Verifies the vector store is reachable. Returns `ready` if all checks pass, `degraded` otherwise.",
    tags=["health"],
)
async def ready():
    checks = {}

    try:
        await store.retrieve(embedding=[0.0] * 1536, query="test")
        checks["vector_store"] = "ok"
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}