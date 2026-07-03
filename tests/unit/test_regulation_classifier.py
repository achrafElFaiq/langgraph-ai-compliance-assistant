"""Unit tests for the regulation classifier node (classify).

This node is a trained scikit-learn multi-label model (no LLM), so it is tested
against the *real* artifacts deterministically. The assertion is the documented
behaviour: a domain question about topic X must flag regulation X.

Labels are compared *representation-insensitively* (case, spaces, underscores):
whether a regulation surfaces as "MiCA", "mica", "AI Act" or "ai_act" is an
incidental encoding detail, not part of the behavioural contract — the identity
of the regulation is.
"""
import pytest

pytestmark = pytest.mark.unit

KNOWN = {"mica", "dora", "aiact", "gdpr"}


def _norm(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "")


def _classify(state_cls, text):
    from src.application.agent.nodes.intent import classify
    return [_norm(r) for r in classify(state_cls(input_text=text))["regulations"]]


def test_classify_returns_known_regulations(state_cls):
    """classify returns a non-empty list drawn only from the known regulation set.

    Matters because retrieval is scoped by this output — an empty or bogus label
    would break downstream retrieval.
    """
    regs = _classify(state_cls, "Question générale de conformité réglementaire européenne.")
    assert regs
    assert set(regs).issubset(KNOWN)


def test_token_offering_question_flags_mica(state_cls):
    """A crypto-token public-offering question flags MiCA — the crypto-asset regulation."""
    regs = _classify(
        state_cls,
        "Nous lançons un jeton utilitaire crypto-actif et souhaitons faire une offre "
        "au public de ce token. Quelles sont nos obligations réglementaires ?",
    )
    assert "mica" in regs


def test_operational_resilience_question_flags_dora(state_cls):
    """An ICT operational-resilience / third-party-provider question flags DORA."""
    regs = _classify(
        state_cls,
        "Quelles sont nos obligations de résilience opérationnelle informatique et de "
        "gestion des risques liés aux prestataires tiers de services TIC pour notre banque ?",
    )
    assert "dora" in regs


def test_ai_system_question_flags_ai_act(state_cls):
    """A high-risk AI-system question flags the AI Act."""
    regs = _classify(
        state_cls,
        "Notre système d'intelligence artificielle de notation de crédit est-il "
        "considéré comme un système d'IA à haut risque au sens du règlement sur l'IA ?",
    )
    assert "aiact" in regs
