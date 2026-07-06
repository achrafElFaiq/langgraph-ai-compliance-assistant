"""Retrieval nodes — ML regulation classifier, hybrid vector/BM25 article retrieval, and fallback search."""

import json
import logging
from datetime import datetime

from src.application.agent.state import State
from src.config.init_embedder import embedder
from src.config.init_store import store

logger = logging.getLogger(__name__)


async def retrieve_articles(state: State) -> dict:
    """Hybrid vector + BM25 retrieval of candidate articles, scoped to the classified regulations."""
    time = datetime.now()

    query = state.input_text
    # Budget scales with how many regulations the classifier flagged: 5 candidates each.
    top_k = 5 * len(state.regulations)
    logger.info("retrieve_articles | started top_k=%d regulations=%s", top_k, state.regulations)

    embedding = await embedder.embed_query(query)
    articles = await store.retrieve(
        embedding=embedding,
        query=query,
        top_k=top_k,
        regulations=state.regulations
    )

    skeleton = {
        a.breadcrumb: {"relevant": None, "excerpts": []}
        for a in articles
    }

    duration = (datetime.now() - time).total_seconds()
    logger.info("retrieve_articles | count=%d top_k=%d regulations=%s duration=%.2fs", len(articles), top_k, state.regulations, duration)
    logger.debug("retrieve_articles | query=%r breadcrumbs=%s duration=%.2fs", query, [a.breadcrumb for a in articles], duration)
    return {
        "retrieved_articles": articles,
        "grounded_skeleton": json.dumps(skeleton, ensure_ascii=False, indent=2)
    }


async def retrieve_fallback(state: State) -> dict:
    """Broader retrieval ignoring the regulation scope, used when grounding finds nothing relevant."""
    logger.warning("retrieve_fallback | fallback triggered — no relevant articles found in scoped retrieval")
    time = datetime.now()

    embedding = await embedder.embed_query(state.input_text)
    articles = await store.retrieve(
        embedding=embedding,
        query=state.input_text,
        top_k=5
    )

    skeleton = {
        a.breadcrumb: {"relevant": None, "excerpts": []}
        for a in articles
    }

    duration = (datetime.now() - time).total_seconds()
    logger.info("retrieve_fallback | count=%d duration=%.2fs", len(articles), duration)
    logger.debug("retrieve_fallback | breadcrumbs=%s duration=%.2fs", [a.breadcrumb for a in articles], duration)
    return {
        "retrieved_articles": articles,
        "grounded_skeleton": json.dumps(skeleton, ensure_ascii=False, indent=2),
        "fallback_attempted": True
    }
