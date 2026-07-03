"""Unit tests for the Pydantic domain/API models — validation and field contracts."""
from datetime import date

import pytest
from pydantic import ValidationError

from src.api.schemas.chat import ChatRequest, ChatResponse, Citation
from src.domain.models.models import (
    Article,
    ArticleChunk,
    EvaluationResult,
    FetchResult,
    StoreResult,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- Article
def test_article_accepts_valid_payload():
    """A fully-specified Article constructs successfully — the happy path must work."""
    art = Article(
        regulation_name="MiCA", title_number=2, chapter_number=1, article_number="4",
        article_title="Offre", breadcrumb="MiCA > T2 > A4", content="text",
        valid_from=date(2024, 12, 30), valid_until=None, source_url="http://x",
    )
    assert art.regulation_name == "MiCA"
    assert art.article_number == "4"


def test_article_missing_required_field_is_rejected():
    """Omitting a required field (regulation_name) must raise — required fields are enforced."""
    with pytest.raises(ValidationError):
        Article(
            title_number=2, chapter_number=1, article_number="4", article_title="t",
            breadcrumb="b", content="c", valid_from=date(2024, 1, 1),
            valid_until=None, source_url="http://x",
        )


def test_article_rejects_unparseable_date():
    """A non-date valid_from must be rejected — type constraints are enforced, not coerced blindly."""
    with pytest.raises(ValidationError):
        Article(
            regulation_name="MiCA", title_number=2, chapter_number=1, article_number="4",
            article_title="t", breadcrumb="b", content="c", valid_from="not-a-date",
            valid_until=None, source_url="http://x",
        )


def test_article_nullable_fields_accept_none():
    """Optional fields (title_number, valid_until) accept None — they are nullable by contract."""
    art = Article(
        regulation_name="GDPR", title_number=None, chapter_number=None, article_number="1",
        article_title=None, breadcrumb="b", content="c", valid_from=date(2018, 5, 25),
        valid_until=None, source_url="http://x",
    )
    assert art.title_number is None
    assert art.valid_until is None


# --------------------------------------------------------------------------- ArticleChunk
def test_article_chunk_embedding_defaults_to_none():
    """embedding is optional and defaults to None — chunks exist before embedding."""
    chunk = ArticleChunk(
        regulation_name="MiCA", title_number=2, chapter_number=1, article_number="4",
        article_title="t", breadcrumb="b", chunk_index=0, chunk_total=1, content="c",
        valid_from=date(2024, 12, 30), valid_until=None, source_url="http://x",
    )
    assert chunk.embedding is None


def test_article_chunk_requires_integer_index():
    """chunk_index must be an integer — a non-numeric value is rejected."""
    with pytest.raises(ValidationError):
        ArticleChunk(
            regulation_name="MiCA", title_number=2, chapter_number=1, article_number="4",
            article_title="t", breadcrumb="b", chunk_index="zero", chunk_total=1, content="c",
            valid_from=date(2024, 12, 30), valid_until=None, source_url="http://x",
        )


# --------------------------------------------------------------------------- Chat schemas
def test_chat_request_requires_input_text():
    """ChatRequest.input_text is required — an empty payload is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest()


def test_chat_request_thread_id_optional():
    """thread_id defaults to None so first-turn requests need only input_text."""
    req = ChatRequest(input_text="hello")
    assert req.thread_id is None


def test_chat_response_defaults():
    """ChatResponse requires answer + thread_id; all richer fields default to empty/zero."""
    resp = ChatResponse(answer="a", thread_id="t")
    assert resp.regulations == []
    assert resp.citations == []
    assert resp.retry_count == 0
    assert resp.fallback_attempted is False


def test_citation_shape():
    """A Citation carries a breadcrumb, a boolean relevance flag, and a list of excerpts."""
    c = Citation(breadcrumb="MiCA > A4", relevant=True, excerpts=["x"])
    assert c.relevant is True
    assert c.excerpts == ["x"]


def test_citation_relevant_must_be_bool_like():
    """relevant must be boolean — a clearly non-boolean string is rejected."""
    with pytest.raises(ValidationError):
        Citation(breadcrumb="b", relevant="maybe", excerpts=[])


# --------------------------------------------------------------------------- Eval / Fetch / Store
def test_evaluation_result_requires_core_metric_lists():
    """The four core metric lists are required; latency/retry fields default to empty."""
    with pytest.raises(ValidationError):
        EvaluationResult(faithfulness=[1.0])  # missing the other required lists
    res = EvaluationResult(
        faithfulness=[1.0], factual_correctness=[1.0],
        context_recall=[1.0], context_precision=[1.0],
    )
    assert res.end_to_end_latency == []
    assert res.node_latencies == {}
    assert res.retry_counts == []


def test_fetch_and_store_result_shapes():
    """FetchResult wraps a list of Articles; StoreResult reports a per-regulation count."""
    fr = FetchResult(articles=[], regulation_name="MiCA", valid_from="2024-12-30", source_url="http://x")
    assert fr.articles == []
    sr = StoreResult(regulation_name="MiCA", article_count=10)
    assert sr.article_count == 10
