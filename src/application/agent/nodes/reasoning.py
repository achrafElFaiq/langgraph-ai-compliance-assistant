"""Reasoning nodes — article grounding, law application, and critic verification loop."""

import asyncio
import json
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from src.application.agent.state import State
from src.config.init_llm import llm, grounder_llm
from src.config.init_llm import critic_llm
from src.config.init_prompts import (
    load_ground_prompt,
    load_apply_prompt
)

logger = logging.getLogger(__name__)


async def ground(state: State) -> dict:
    """For each retrieved article, judge relevance to the question and extract the exact supporting excerpts."""
    logger.info("ground | started articles=%d", len(state.retrieved_articles))
    time = datetime.now()

    ground_prompt = load_ground_prompt()
    skeleton_init = json.loads(state.grounded_skeleton)

    # Grounding one article
    async def _ground_article(article):
        response = await grounder_llm.ainvoke([
            SystemMessage(content=ground_prompt),
            HumanMessage(content=f"Question: {state.input_text}\n\nArticle ({article.breadcrumb}):\n{article.content}")
        ])
        raw = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not raw:
            return article.breadcrumb, {"relevant": False, "excerpts": []}
        try:
            parsed = json.loads(raw.group())
            # ensure excerpts is always a list
            if parsed.get("relevant") and not isinstance(parsed.get("excerpts"), list):
                parsed["excerpts"] = [parsed["excerpts"]] if parsed.get("excerpts") else []
        except json.JSONDecodeError:
            parsed = {"relevant": False, "excerpts": []}

        return article.breadcrumb, parsed

    results = await asyncio.gather(*[_ground_article(a) for a in state.retrieved_articles])

    skeleton = {}
    for breadcrumb, slots in results:
        if breadcrumb not in skeleton_init:
            logger.warning("ground | unexpected breadcrumb '%s', skipping", breadcrumb)
            continue
        skeleton[breadcrumb] = slots

    for breadcrumb in skeleton_init:
        if breadcrumb not in skeleton:
            logger.warning("ground | missing article '%s', using fallback", breadcrumb)
            skeleton[breadcrumb] = {"relevant": False, "excerpts": []}

    relevant_count = sum(1 for v in skeleton.values() if v.get("relevant") is True)
    duration = (datetime.now() - time).total_seconds()

    logger.info("ground | finished relevant=%d/%d duration=%.2fs", relevant_count, len(skeleton), duration)
    if relevant_count == 0:
        logger.warning("ground | zero relevant articles found")
    logger.debug(
        "ground | relevance=%s duration=%.2fs",
        {bc: v.get("relevant") for bc, v in skeleton.items()},
        duration,
    )

    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=2)
    return {"grounded_skeleton": skeleton_json}


async def apply(state: State) -> dict:
    """For each relevant article, analyse how it applies to the user's situation and surface any gaps."""
    logger.info("apply | started")
    time = datetime.now()

    skeleton = json.loads(state.grounded_skeleton)
    relevant = {k: v for k, v in skeleton.items() if v.get("relevant") is True}

    if not relevant:
        logger.info("apply | finished relevant=0 (no relevant articles)")
        empty = json.dumps({"findings": []}, ensure_ascii=False)
        return {"apply_output": empty}

    apply_prompt = load_apply_prompt()

    async def _apply_article(breadcrumb: str, data: dict) -> dict:
        excerpts_text = "\n".join([f"- {e}" for e in data["excerpts"]])

        response = await llm.ainvoke([
            SystemMessage(content=apply_prompt),
            HumanMessage(content=(
                f"Question: {state.input_text}\n\n"
                f"Article: {breadcrumb}\n"
                f"Extraits:\n{excerpts_text}"
            ))
        ])

        raw = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not raw:
            return {"article": breadcrumb, "excerpts_used": data["excerpts"], "finding": ""}
        try:
            parsed = json.loads(raw.group())
            parsed["article"] = breadcrumb
            parsed["excerpts_used"] = data["excerpts"]
            return parsed
        except json.JSONDecodeError:
            return {"article": breadcrumb, "excerpts_used": data["excerpts"], "finding": response.content.strip()}

    findings = list(await asyncio.gather(*[
        _apply_article(bc, data) for bc, data in relevant.items()
    ]))

    apply_output = json.dumps(
        {"findings": findings},
        ensure_ascii=False,
        indent=2
    )

    logger.info("apply | finished relevant=%d duration=%.2fs", len(relevant), (datetime.now() - time).total_seconds())
    return {"apply_output": apply_output}


async def critic_answer(state: State) -> dict:
    """Verify the drafted answer against the findings, flagging unsupported claims to trigger a revision."""
    logger.info("critic_answer | started")
    time = datetime.now()

    apply_data = json.loads(state.apply_output)
    findings = apply_data.get("findings", [])

    # Build reference text from findings only
    findings_text = "\n\n".join([
        f"{f['article']}:\n"
        f"Extraits: {json.dumps(f['excerpts_used'], ensure_ascii=False)}\n"
        f"Conclusion: {f['finding']}"
        for f in findings
    ])

    # Step 1: extract claims from answer
    try:
        claims_response = await critic_llm.ainvoke([
            SystemMessage(content=(
                "Extrayez chaque affirmation factuelle de la réponse sous forme de liste. "
                "Une affirmation par ligne. Phrases courtes uniquement. Pas de numérotation."
            )),
            HumanMessage(content=state.answer)
        ])
        claims = [
            line.strip().lstrip("0123456789.-) ")
            for line in claims_response.content.strip().split("\n")
            if line.strip()
        ]
    except Exception as e:
        logger.debug("critic_answer | claim_extraction_error=%s", e)
        claims = []

    # Step 2: check each claim against findings in parallel
    async def _check_claim(claim: str) -> tuple[str, bool]:
        try:
            response = await critic_llm.ainvoke([
                SystemMessage(content=(
                    "Répondez uniquement par OUI ou NON. "
                    "Cette affirmation est-elle directement soutenue par l'une des conclusions fournies ? "
                    "Une inférence directe et logique des conclusions compte comme soutenue. "
                    "Répondez NON uniquement si l'affirmation n'a aucune base dans les conclusions."
                )),
                HumanMessage(content=(
                    f"Affirmation : \"{claim}\"\n\n"
                    f"Conclusions:\n{findings_text}"
                ))
            ])
            supported = response.content.strip().upper().startswith("OUI")
            return claim, supported
        except Exception as e:
            logger.debug("critic_answer | claim_check_error=%s", e)
            return claim, True

    claim_results = await asyncio.gather(*[_check_claim(c) for c in claims]) if claims else []
    invented = [claim for claim, supported in claim_results if not supported]

    duration = (datetime.now() - time).total_seconds()

    # Approved
    if not invented:
        logger.info("critic_answer | approved duration=%.2fs", duration)
        return {"critic_opinion": ""}

    # Build accumulated feedback
    logger.warning("critic_answer | rejected invented_claims=%d retry=%d duration=%.2fs", len(invented), state.retry_count, duration)
    logger.debug("critic_answer | invented=%s", invented)

    feedback_lines = [f"- \"{claim}\" [IN ANSWER, NOT IN SKELETON]" for claim in invented]
    feedback = "\n".join(feedback_lines)

    if state.critic_opinion:
        accumulated = state.critic_opinion + f"\n\n[Round {state.retry_count + 1}]:\n{feedback}"
    else:
        accumulated = f"[Round 0]:\n{feedback}"

    return {"critic_opinion": accumulated}
