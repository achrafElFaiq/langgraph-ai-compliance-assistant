"""Unit tests for LangGraph node functions.

Contract asserted for every node (per the task spec):
  * returns a plain dict that is a valid *partial* State update (keys ⊆ State fields),
  * does not mutate its input State,
  * produces the documented output for a crafted input.

All LLM dependencies are mocked at the node module's injection point.
"""
import json

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- classify_intent
async def test_classify_intent_synthesis_shortcut_skips_llm(
    state_cls, exploding_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """The 'Generate synthesis' trigger routes to synthesis WITHOUT calling the LLM."""
    import src.application.agent.nodes.intent as intent
    monkeypatch.setattr(intent, "llm", exploding_llm)
    state = state_cls(input_text="Generate synthesis")
    check = snapshot_guard(state)

    out = await intent.classify_intent(state)

    assert out["route"] == "synthesis"
    assert exploding_llm.calls == []  # LLM must not be consulted for the shortcut
    assert_valid_update(out)
    check()


async def test_classify_intent_maps_model_label_to_route(
    state_cls, make_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """A message classified as 'chitchat' by the intent model routes to chitchat."""
    import src.application.agent.nodes.intent as intent
    monkeypatch.setattr(intent, "llm", make_llm(content="chitchat"))
    state = state_cls(input_text="bonjour !")
    check = snapshot_guard(state)

    out = await intent.classify_intent(state)

    assert out["route"] == "chitchat"
    assert_valid_update(out)
    check()


async def test_classify_intent_normalises_unknown_label(state_cls, make_llm, monkeypatch):
    """An unrecognised classification must fall back to a valid route, never leak garbage.

    Reasonable expectation: the node guarantees route ∈ {research, followup, chitchat}
    regardless of noisy model output.
    """
    import src.application.agent.nodes.intent as intent
    monkeypatch.setattr(intent, "llm", make_llm(content="banana-42"))
    out = await intent.classify_intent(state_cls(input_text="anything"))
    assert out["route"] in {"research", "followup", "chitchat"}


# --------------------------------------------------------------------------- direct_answer
async def test_direct_answer_returns_answer_and_messages(
    state_cls, make_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """direct_answer returns the model's reply as `answer` and appends to `messages`."""
    import src.application.agent.nodes.generation as generation
    monkeypatch.setattr(generation, "llm", make_llm(content="Bonjour, comment puis-je aider ?"))
    state = state_cls(input_text="salut", messages=[])
    check = snapshot_guard(state)

    out = await generation.direct_answer(state)

    assert out["answer"] == "Bonjour, comment puis-je aider ?"
    assert "messages" in out and isinstance(out["messages"], list)
    assert_valid_update(out)
    check()


# --------------------------------------------------------------------------- answer
async def test_answer_composes_from_findings(
    state_cls, make_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """answer drafts a reply from the applied findings and returns answer + retry_count."""
    import src.application.agent.nodes.generation as generation
    monkeypatch.setattr(generation, "llm", make_llm(content="Voici la réponse finale."))
    apply_output = json.dumps({
        "findings": [{"article": "MiCA > A4", "excerpts_used": ["e"], "finding": "f"}],
        "gaps": [],
    })
    state = state_cls(input_text="q", apply_output=apply_output, messages=[])
    check = snapshot_guard(state)

    out = await generation.answer(state)

    assert out["answer"] == "Voici la réponse finale."
    assert "retry_count" in out
    assert_valid_update(out)
    check()


# --------------------------------------------------------------------------- synthesize
async def test_synthesize_returns_final_report(
    state_cls, make_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """synthesize returns the generated report under `final_report`."""
    import src.application.agent.nodes.generation as generation
    monkeypatch.setattr(generation, "llm", make_llm(content="# Rapport de conformité"))
    state = state_cls(input_text="Generate synthesis", messages=[])
    check = snapshot_guard(state)

    out = await generation.synthesize(state)

    assert out["final_report"] == "# Rapport de conformité"
    assert_valid_update(out)
    check()


# --------------------------------------------------------------------------- ground
async def test_ground_produces_skeleton_keyed_by_breadcrumb(
    state_cls, sample_article, make_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """ground returns a grounded_skeleton JSON keyed by each article's breadcrumb."""
    import src.application.agent.nodes.reasoning as reasoning
    monkeypatch.setattr(
        reasoning, "grounder_llm",
        make_llm(content='{"relevant": true, "excerpts": ["exact quote"]}'),
    )
    skeleton_init = json.dumps({sample_article.breadcrumb: {"relevant": None, "excerpts": []}})
    state = state_cls(
        input_text="q", retrieved_articles=[sample_article], grounded_skeleton=skeleton_init,
    )
    check = snapshot_guard(state)

    out = await reasoning.ground(state)

    assert "grounded_skeleton" in out
    parsed = json.loads(out["grounded_skeleton"])
    assert sample_article.breadcrumb in parsed
    assert_valid_update(out)
    check()


# --------------------------------------------------------------------------- apply
async def test_apply_with_no_relevant_articles_skips_llm(
    state_cls, exploding_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """When nothing was judged relevant, apply returns empty findings and never calls the LLM."""
    import src.application.agent.nodes.reasoning as reasoning
    monkeypatch.setattr(reasoning, "llm", exploding_llm)
    skeleton = json.dumps({"MiCA > A4": {"relevant": False, "excerpts": []}})
    state = state_cls(input_text="q", grounded_skeleton=skeleton)
    check = snapshot_guard(state)

    out = await reasoning.apply(state)

    assert exploding_llm.calls == []
    parsed = json.loads(out["apply_output"])
    assert parsed.get("findings") == []
    assert_valid_update(out)
    check()
