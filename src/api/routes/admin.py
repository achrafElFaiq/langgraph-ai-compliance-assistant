import logging

from fastapi import APIRouter

from src.config.init_regulations import REGULATIONS
from src.config.init_store import store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/ingestion/stats",
    summary="Ingestion statistics",
    description="Returns the number of stored articles per regulation and the total.",
    tags=["ingestion"],
)
async def ingestion_stats():
    counts = await store.count_articles_by_regulation()

    # Report every configured regulation, defaulting to 0 when nothing is stored.
    regulations = []
    for alias, meta in REGULATIONS.items():
        name = meta.get("regulation_name", alias)
        regulations.append({"name": name, "articles": counts.get(name, 0)})

    return {
        "regulations": regulations,
        "total_articles": sum(r["articles"] for r in regulations),
    }
