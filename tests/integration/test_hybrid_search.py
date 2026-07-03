"""Integration test for hybrid BM25 + vector retrieval (Reciprocal Rank Fusion).

Real DB, no LLM. Requires TEST_DATABASE_URL/DATABASE_URL; skips otherwise.
"""
from datetime import date

import pytest

from src.domain.models.models import Article, ArticleChunk

pytestmark = pytest.mark.integration

EMBED_DIM = 1536


def _pair(num, breadcrumb, content, embedding):
    art = Article(
        regulation_name="MiCA", title_number=2, chapter_number=1, article_number=num,
        article_title="t", breadcrumb=breadcrumb, content=content,
        valid_from=date(2024, 12, 30), valid_until=None, source_url=f"http://t/{num}",
    )
    chunk = ArticleChunk(
        regulation_name="MiCA", title_number=2, chapter_number=1, article_number=num,
        article_title="t", breadcrumb=breadcrumb, chunk_index=0, chunk_total=1,
        content=content, valid_from=date(2024, 12, 30), valid_until=None,
        source_url=f"http://t/{num}", embedding=embedding,
    )
    return art, chunk


async def test_exact_keyword_match_ranks_first(repo):
    """An article whose text matches the query's exact keywords ranks first.

    Both articles share the same embedding so the vector arm is neutral; RRF must
    let the BM25 (keyword) arm decide, surfacing the keyword-matching article on top.
    Matters because pure vector search alone can miss exact-term relevance.
    """
    emb = [0.05] * EMBED_DIM  # identical → vector arm cannot distinguish the two
    target_bc = "MiCA > Titre 5 > Article 60"
    other_bc = "MiCA > Titre 9 > Article 99"

    target = _pair("60", target_bc,
                   "notification de passeport à l'autorité compétente pour prestataire", emb)
    other = _pair("99", other_bc,
                  "dispositions générales diverses sans rapport particulier", emb)

    ids = await repo.store_articles([target[0], other[0]])
    await repo.store_chunks([target[1], other[1]], ids)

    results = await repo.retrieve(
        embedding=emb,
        query="notification passeport autorité compétente prestataire",
        top_k=5,
    )

    assert results, "no results returned"
    assert results[0].breadcrumb == target_bc, (
        f"keyword-matching article did not rank first; got {[a.breadcrumb for a in results]}"
    )
