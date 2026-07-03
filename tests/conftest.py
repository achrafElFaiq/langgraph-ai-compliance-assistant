"""Shared fixtures for the compliance-assistant test suite.

Design rules honoured here:
  * Tests target the *public contract* (inputs, outputs, side effects, documented
    behaviour) — never the internals of a method body.
  * All external dependencies (LLM, DB, HTTP) are mockable via seams provided here.
"""
import os

# Dummy credentials so importing src modules never fails offline (LLM/Langfuse
# clients are constructed at import time). These are never used to reach a network.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-test")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3999")

import copy
import dataclasses

import httpx
import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# LLM doubles
# ---------------------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Stand-in for a LangChain chat model.

    Configure with exactly one of:
      * ``content``   — a single string returned for every call
      * ``responses`` — a list consumed in call order
      * ``handler``   — a callable ``(messages) -> str``
    Every invocation is recorded in ``.calls`` so tests can assert call counts.
    """

    def __init__(self, content=None, responses=None, handler=None):
        self._content = content
        self._responses = list(responses) if responses is not None else None
        self._handler = handler
        self.calls = []

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        if self._handler is not None:
            return _FakeMessage(self._handler(messages))
        if self._responses is not None:
            return _FakeMessage(self._responses.pop(0) if self._responses else "")
        return _FakeMessage(self._content if self._content is not None else "")


class ExplodingLLM:
    """LLM double that fails if invoked — asserts a path must NOT call the model."""

    def __init__(self):
        self.calls = []

    async def ainvoke(self, *args, **kwargs):
        self.calls.append(args)
        raise AssertionError("LLM was called on a path that must not call it")


@pytest.fixture
def make_llm():
    """Factory returning configurable FakeLLM instances."""
    def _make(content=None, responses=None, handler=None):
        return FakeLLM(content=content, responses=responses, handler=handler)
    return _make


@pytest.fixture
def exploding_llm():
    """An LLM double that raises if called."""
    return ExplodingLLM()


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def state_cls():
    from src.application.agent.state import State
    return State


@pytest.fixture
def state_field_names():
    """The set of valid State field names — a node's return dict must be a subset."""
    from src.application.agent.state import State
    return {f.name for f in dataclasses.fields(State)}


@pytest.fixture
def assert_valid_update(state_field_names):
    """Assert a node returned a valid partial-State update: a dict whose every key
    is a real State field (no extra/unknown keys)."""
    def _assert(result):
        assert isinstance(result, dict), f"node must return a dict, got {type(result)!r}"
        unknown = set(result) - state_field_names
        assert not unknown, f"node returned keys not present on State: {unknown}"
    return _assert


@pytest.fixture
def snapshot_guard():
    """Return a factory that snapshots a State and yields a checker asserting it
    was not mutated by the node under test."""
    def _make(state):
        before = copy.deepcopy(state)

        def _check():
            assert state == before, "node mutated its input State (must be pure)"
        return _check
    return _make


# ---------------------------------------------------------------------------
# Sample domain objects
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_article():
    from datetime import date
    from src.domain.models.models import Article
    return Article(
        regulation_name="MiCA",
        title_number=2,
        chapter_number=1,
        article_number="4",
        article_title="Offre au public",
        breadcrumb="MiCA > Titre 2 > Article 4",
        content="Une personne ne peut offrir au public un crypto-actif que sous conditions.",
        valid_from=date(2024, 12, 30),
        valid_until=None,
        source_url="https://example.test/mica/4",
    )


# ---------------------------------------------------------------------------
# FastAPI app / async client
# ---------------------------------------------------------------------------
@pytest.fixture
def app():
    from src.api.app import app as fastapi_app
    return fastapi_app


@pytest_asyncio.fixture
async def client(app):
    """Async client over the ASGI app.

    Note: httpx's ASGITransport does NOT run the app lifespan, so the DB stays
    unconnected unless a test wires it explicitly — exactly what we want for
    controlled endpoint tests.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Database (integration)
# ---------------------------------------------------------------------------
def _db_url():
    return os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")


@pytest_asyncio.fixture
async def repo():
    """A connected PostgresRegulationRepository against the test DB.

    Skips when no DB is configured/reachable. Clears the store before and after
    so tests are isolated.
    """
    url = _db_url()
    if not url:
        pytest.skip("Set TEST_DATABASE_URL or DATABASE_URL to run DB integration tests")
    from src.infrastructure.store.postgres_store import PostgresRegulationRepository
    r = PostgresRegulationRepository(connection_string=url)
    try:
        await r.connect()
    except Exception as e:  # pragma: no cover - env dependent
        pytest.skip(f"Test database not reachable: {e}")
    try:
        await r.clear()
        yield r
    finally:
        try:
            await r.clear()
        finally:
            await r.close()


@pytest_asyncio.fixture
async def live_store(monkeypatch):
    """Connect a store to the test DB and inject it wherever routes reference the
    singleton, so endpoints see a live DB connection. Skips if no DB.
    """
    url = _db_url()
    if not url:
        pytest.skip("Set TEST_DATABASE_URL or DATABASE_URL to run DB integration tests")
    from src.infrastructure.store.postgres_store import PostgresRegulationRepository
    store = PostgresRegulationRepository(connection_string=url)
    try:
        await store.connect()
    except Exception as e:  # pragma: no cover - env dependent
        pytest.skip(f"Test database not reachable: {e}")

    import src.config.init_store as init_store
    import src.api.routes.health as health_mod
    import src.api.routes.admin as admin_mod
    monkeypatch.setattr(init_store, "store", store, raising=False)
    monkeypatch.setattr(health_mod, "store", store, raising=False)
    monkeypatch.setattr(admin_mod, "store", store, raising=False)
    try:
        yield store
    finally:
        await store.close()
