"""End-to-end tests against a running stack (``docker compose up``), real LLM.

Base URL comes from E2E_BASE_URL (default http://localhost:8000). Every test skips
if the stack is unreachable. These assert user-visible behaviour, not internals.

Assumes the DB has been ingested (``docker compose --profile jobs run --rm jobs
main.py index``) so retrieval has content.
"""
import json
import os

import httpx
import pytest

pytestmark = pytest.mark.e2e

BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(120.0)


@pytest.fixture(scope="module")
def base_url():
    """Skip the whole e2e module unless the stack answers /health."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"Stack not reachable at {BASE_URL}: {e}")
    return BASE_URL


def _stream_done(base_url, input_text, thread_id=None):
    """POST /chat/stream and return the parsed final 'done' event payload."""
    payload = {"input_text": input_text, "thread_id": thread_id}
    done = None
    with httpx.stream("POST", f"{base_url}/chat/stream", json=payload, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        buffer = ""
        for chunk in resp.iter_text():
            buffer += chunk
        for line in buffer.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try:
                    obj = json.loads(line[len("data:"):].strip())
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "done":
                    done = obj
    assert done is not None, "stream never emitted a 'done' event"
    return done


def test_mica_question_cites_an_article(base_url):
    """A MiCA compliance question yields an answer that cites at least one article.

    Matters because uncited compliance advice is unusable — grounding is the point.
    """
    done = _stream_done(
        base_url,
        "Nous lançons un token utilitaire en France ; quelles sont nos obligations "
        "avant l'offre au public sous MiCA ?",
    )
    answer = (done.get("answer") or done.get("final_report") or "")
    citations = done.get("citations") or []
    relevant_citations = [c for c in citations if c.get("relevant")]
    assert relevant_citations or "article" in answer.lower(), (
        "no article citation surfaced for a MiCA compliance question"
    )


def test_followup_uses_prior_context(base_url):
    """A follow-up in the same thread produces a substantive answer without restarting.

    Reasonable expectation (context retention is fuzzy over a real LLM): the reply is
    non-trivial and the thread id is preserved, i.e. the turn continues rather than
    re-introducing itself.
    """
    first = _stream_done(
        base_url,
        "Notre banque recourt à un prestataire cloud tiers ; quelles obligations sous DORA ?",
    )
    thread_id = first.get("thread_id")
    assert thread_id, "no thread_id returned on first turn"

    follow = _stream_done(base_url, "Et si le prestataire est situé hors de l'UE ?", thread_id=thread_id)
    assert follow.get("thread_id") == thread_id
    assert len((follow.get("answer") or follow.get("final_report") or "").strip()) > 40


def test_greeting_returns_short_answer_without_citations(base_url):
    """A greeting gets a short conversational reply with no article citations.

    Matters because small talk must not trigger the retrieval/grounding pipeline.
    """
    done = _stream_done(base_url, "Bonjour, merci de votre aide !")
    citations = [c for c in (done.get("citations") or []) if c.get("relevant")]
    assert not citations, "a greeting should not produce article citations"


def test_synthesis_after_two_research_turns(base_url):
    """After ≥2 research turns, requesting a synthesis returns a structured report.

    Matters because the synthesis is the deliverable artifact of a session.
    """
    tid = _stream_done(
        base_url, "Quelles obligations MiCA pour l'émission d'un stablecoin ?",
    ).get("thread_id")
    _stream_done(base_url, "Et quelles obligations DORA pour notre infrastructure ?", thread_id=tid)

    done = _stream_done(base_url, "Generate synthesis", thread_id=tid)
    report = done.get("final_report") or ""
    assert len(report.strip()) > 100, "synthesis report is empty or trivially short"


def test_health_ready_is_200_when_up(base_url):
    """/health/ready returns 200 once all services are up — the readiness contract."""
    r = httpx.get(f"{base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200
