"""Integration tests for the /chat endpoint.

The agent (compiled_graph) is mocked as a collaborator so we test the endpoint's
own contract — request validation and response shape — without a real LLM/DB.
"""
import pytest

from src.api.schemas.chat import ChatResponse

pytestmark = pytest.mark.integration


class _FakeGraph:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, inputs, config=None):
        return self._result


async def test_chat_response_matches_documented_schema(client, monkeypatch):
    """POST /chat returns a body that validates against ChatResponse and echoes the answer.

    Matters because clients depend on the documented response schema; a shape drift
    breaks them silently.
    """
    import src.api.routes.chat as chat_mod
    monkeypatch.setattr(chat_mod, "compiled_graph", _FakeGraph({"answer": "Une réponse de conformité."}))

    r = await client.post("/chat", json={"input_text": "Faut-il un agrément ?"})

    assert r.status_code == 200
    ChatResponse.model_validate(r.json())  # raises if the shape is wrong
    assert r.json()["answer"] == "Une réponse de conformité."
    assert r.json()["thread_id"]  # a thread id is always assigned


async def test_chat_preserves_supplied_thread_id(client, monkeypatch):
    """A supplied thread_id is echoed back — conversation continuity depends on it."""
    import src.api.routes.chat as chat_mod
    monkeypatch.setattr(chat_mod, "compiled_graph", _FakeGraph({"answer": "ok"}))

    r = await client.post("/chat", json={"input_text": "suite", "thread_id": "thread-xyz"})

    assert r.status_code == 200
    assert r.json()["thread_id"] == "thread-xyz"


async def test_chat_rejects_missing_input_text(client):
    """POST /chat without input_text is rejected (422) — required-field validation holds."""
    r = await client.post("/chat", json={})
    assert r.status_code == 422
