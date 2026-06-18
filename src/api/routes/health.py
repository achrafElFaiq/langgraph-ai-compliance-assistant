from fastapi import APIRouter
from src.config.init_store import store
from src.config.init_llm import llm

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/health/ready")
async def ready():
    checks = {}

    try:
        await store.retrieve(embedding=[0.0] * 1536, query="test")
        checks["vector_store"] = "ok"
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"

    try:
        llm.invoke("ping")
        checks["llm"] = "ok"
    except Exception as e:
        checks["llm"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}