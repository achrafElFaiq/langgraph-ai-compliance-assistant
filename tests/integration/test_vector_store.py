"""Integration tests for the pgvector-backed store — real DB, no LLM.

Requires TEST_DATABASE_URL (or DATABASE_URL) pointing at a pgvector database with
the project schema (db/init.sql). Skips otherwise.
"""
from datetime import date

import pytest

from src.domain.models.models import Article, ArticleChunk

pytestmark = pytest.mark.integration

EMBED_DIM = 1536


def _article(num, breadcrumb, content, reg="MiCA"):
    return Article(
        regulation_name=reg, title_number=2, chapter_number=1, article_number=num,
        article_title="t", breadcrumb=breadcrumb, content=content,
        valid_from=date(2024, 12, 30), valid_until=None, source_url=f"http://t/{num}",
    )


def _chunk(num, breadcrumb, content, embedding, reg="MiCA"):
    return ArticleChunk(
        regulation_name=reg, title_number=2, chapter_number=1, article_number=num,
        article_title="t", breadcrumb=breadcrumb, chunk_index=0, chunk_total=1,
        content=content, valid_from=date(2024, 12, 30), valid_until=None,
        source_url=f"http://t/{num}", embedding=embedding,
    )


async def test_insert_then_retrieve_by_similarity(repo):
    """A chunk stored with an embedding is retrievable by that same embedding,
    and comes back with its correct breadcrumb + regulation metadata.

    Matters because similarity retrieval feeding the wrong metadata would poison
    every downstream grounding/citation.
    """
    breadcrumb = "MiCA > Titre 2 > Article 4"
    emb = [0.1] * EMBED_DIM
    ids = await repo.store_articles([_article("4", breadcrumb, "offre au public de crypto-actifs")])
    await repo.store_chunks([_chunk("4", breadcrumb, "offre au public de crypto-actifs", emb)], ids)

    results = await repo.retrieve(embedding=emb, query="offre au public", top_k=5)

    match = [a for a in results if a.breadcrumb == breadcrumb]
    assert match, f"stored article not retrieved; got {[a.breadcrumb for a in results]}"
    assert match[0].regulation_name == "MiCA"


async def test_retrieve_respects_top_k(repo):
    """retrieve returns no more than top_k articles — the budget is honoured."""
    emb = [0.2] * EMBED_DIM
    arts, chunks = [], []
    for i in range(4):
        bc = f"MiCA > Titre 2 > Article {i}"
        arts.append(_article(str(i), bc, f"contenu réglementaire numéro {i}"))
        chunks.append(_chunk(str(i), bc, f"contenu réglementaire numéro {i}", emb))
    ids = await repo.store_articles(arts)
    await repo.store_chunks(chunks, ids)

    results = await repo.retrieve(embedding=emb, query="contenu réglementaire", top_k=2)

    assert len(results) <= 2
