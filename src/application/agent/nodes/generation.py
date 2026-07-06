"""Generation nodes — answer drafting, direct chitchat responses, and synthesis report generation."""

import json
import logging
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.application.agent.state import State
from src.config.init_llm import llm
from src.config.init_prompts import (
    load_answer_prompt,
    load_synthesis_prompt,
)

logger = logging.getLogger(__name__)


async def direct_answer(state: State) -> dict:
    """Answer chitchat / off-topic messages conversationally, without retrieval."""
    logger.info("direct_answer | started")

    response = await llm.ainvoke([
        SystemMessage(content=(
            "Vous êtes un assistant de conformité réglementaire européenne. "
            "Répondez de manière naturelle et concise."
        )),
        *state.messages,
        HumanMessage(content=state.input_text)
    ])

    logger.info("direct_answer | finished")

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state.input_text),
            AIMessage(content=response.content)
        ]
    }


async def answer(state: State) -> dict:
    """Compose the final grounded answer from the applied findings, incorporating any critic feedback."""
    logger.info("answer | started retry=%d", state.retry_count)
    time = datetime.now()

    apply_data = json.loads(state.apply_output)
    findings = apply_data.get("findings", [])
    gaps = apply_data.get("gaps", [])

    critic_section = ""
    retry_count = state.retry_count
    if state.critic_opinion:
        logger.debug("answer | using critic feedback retry=%d feedback=%r", state.retry_count, state.critic_opinion[:200])
        critic_section = (
            f"\n\nCette réponse a déjà été vérifiée. "
            f"Corrigez les points suivants sans exception:\n{state.critic_opinion}"
        )
        retry_count += 1

    findings_text = json.dumps(findings, ensure_ascii=False, indent=2)
    gaps_text = json.dumps(gaps, ensure_ascii=False, indent=2)

    response = await llm.ainvoke([
        SystemMessage(content=load_answer_prompt()),
        *state.messages,
        HumanMessage(content=(
            f"Question: {state.input_text}\n\n"
            f"Conclusions:\n{findings_text}\n\n"
            f"Informations manquantes:\n{gaps_text}"
            f"{critic_section}"
        ))
    ])

    duration = (datetime.now() - time).total_seconds()
    logger.info("answer | finished duration=%.2fs", duration)
    return {
        "answer": response.content,
        "retry_count": retry_count,
        "messages": [
            HumanMessage(content=state.input_text),
            AIMessage(content=response.content)
        ]
    }


async def synthesize(state: State) -> dict:
    """Generate the end-of-conversation compliance synthesis report."""
    logger.info("synthesize | started")
    time = datetime.now()

    response = await llm.ainvoke([
        SystemMessage(content=load_synthesis_prompt()),
        *state.messages,
        HumanMessage(content="Generate the compliance report based on our conversation.")
    ])

    logger.info("synthesize | finished duration=%.2fs", (datetime.now() - time).total_seconds())
    return {"final_report": response.content}
