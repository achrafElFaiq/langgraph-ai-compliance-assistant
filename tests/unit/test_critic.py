"""Unit tests for the critic node (critic_answer).

Contract: the critic verifies the drafted answer against the applied findings.
  * every claim supported  → approves  (empty critic_opinion)
  * a claim unsupported    → rejects   (non-empty critic_opinion, driving a revision)

The critic LLM is mocked so we control the "supported / not supported" verdict and
assert the node's accept/reject contract, not the model's judgement.
"""
import json

import pytest

pytestmark = pytest.mark.unit


def _state_with_answer(state_cls):
    apply_output = json.dumps({
        "findings": [{
            "article": "MiCA > Titre 2 > Article 4",
            "excerpts_used": ["Une personne ne peut offrir au public un crypto-actif que sous conditions."],
            "finding": "Une offre au public requiert un livre blanc.",
        }],
        "gaps": [],
    })
    return state_cls(
        input_text="q",
        answer="Une offre au public de crypto-actif requiert un livre blanc.",
        apply_output=apply_output,
        retry_count=0,
    )


async def test_critic_accepts_supported_answer(
    state_cls, make_llm, monkeypatch, assert_valid_update, snapshot_guard
):
    """When every claim is supported by the findings, the critic approves (empty opinion).

    Matters because approval is the loop's exit condition — a false rejection would
    spin the answer↔critic loop needlessly.
    """
    import src.application.agent.nodes.reasoning as reasoning
    monkeypatch.setattr(reasoning, "critic_llm", make_llm(handler=lambda _m: "OUI"))
    state = _state_with_answer(state_cls)
    check = snapshot_guard(state)

    out = await reasoning.critic_answer(state)

    assert out["critic_opinion"] == ""
    assert_valid_update(out)
    check()


async def test_critic_rejects_unsupported_answer(
    state_cls, make_llm, monkeypatch, assert_valid_update
):
    """When a claim is not supported by the findings, the critic rejects (non-empty opinion).

    Matters because this is the safety net against hallucinated, uncited claims.
    """
    import src.application.agent.nodes.reasoning as reasoning
    monkeypatch.setattr(reasoning, "critic_llm", make_llm(handler=lambda _m: "NON"))
    state = _state_with_answer(state_cls)

    out = await reasoning.critic_answer(state)

    assert out["critic_opinion"] != ""
    assert isinstance(out["critic_opinion"], str)
