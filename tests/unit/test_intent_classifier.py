"""Unit tests for the intent classifier node (classify_intent).

The LLM is mocked, so these assert the node's *contract*: it faithfully translates
the intent model's one-word classification into ``state.route``. Real-world
classification quality is covered by the e2e layer.
"""
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "message, model_label, expected_route",
    [
        ("Devons-nous obtenir un agrément pour émettre un token ?", "research", "research"),
        ("Bonjour, merci beaucoup !", "chitchat", "chitchat"),
        ("Peux-tu reformuler ta réponse précédente ?", "followup", "followup"),
    ],
)
async def test_intent_routes_by_classification(
    state_cls, make_llm, monkeypatch, message, model_label, expected_route
):
    """A compliance question → research, a greeting → chitchat, a follow-up → followup.

    Matters because ``route`` drives which branch of the graph runs; a wrong route
    sends the user down the wrong pipeline.
    """
    import src.application.agent.nodes.intent as intent
    monkeypatch.setattr(intent, "llm", make_llm(content=model_label))

    out = await intent.classify_intent(state_cls(input_text=message))

    assert out["route"] == expected_route


async def test_research_route_resets_iteration_state(state_cls, make_llm, monkeypatch):
    """Routing to 'research' returns only valid State keys (a fresh research turn).

    Matters because a new question must not inherit a stale retry/critic loop state.
    """
    import src.application.agent.nodes.intent as intent
    from src.application.agent.state import State
    import dataclasses

    monkeypatch.setattr(intent, "llm", make_llm(content="research"))
    out = await intent.classify_intent(state_cls(input_text="nouvelle question de conformité"))

    assert out["route"] == "research"
    valid = {f.name for f in dataclasses.fields(State)}
    assert set(out).issubset(valid)
