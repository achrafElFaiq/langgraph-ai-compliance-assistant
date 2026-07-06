from fastapi import APIRouter, Response

from src.config.init_store import store

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
    description="Verifies the vector store is reachable. Returns 200/`ready` when all checks pass, 503/`degraded` otherwise.",
    tags=["health"],
)
async def ready(response: Response):
    checks = {}

    try:
        await store.retrieve(embedding=[0.0] * 1536, query="test")
        checks["vector_store"] = "ok"
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        # Signal "not ready" so orchestrators stop routing traffic to this instance.
        response.status_code = 503
    return {"status": "ready" if all_ok else "degraded", "checks": checks}