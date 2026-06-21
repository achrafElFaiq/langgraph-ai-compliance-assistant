import asyncio
import json
import os
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.application.agent.state import State
from src.config.init_llm import llm, grounder_llm
from src.config.init_llm import critic_llm
from src.config.init_prompts import (
    load_ground_prompt,
    load_apply_prompt
)

# For each retrieved article it produces if it's relevant to the question and if so it adds the exact excrept
async def ground(state: State) -> dict:
    print("[Started Node] ground")
    os.makedirs("run", exist_ok=True)
    os.makedirs("datasets/skeletons", exist_ok=True)
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
            print(f"[ground] WARNING: unexpected breadcrumb '{breadcrumb}', skipping")
            continue
        skeleton[breadcrumb] = slots

    for breadcrumb in skeleton_init:
        if breadcrumb not in skeleton:
            print(f"[ground] WARNING: missing article '{breadcrumb}', using fallback")
            skeleton[breadcrumb] = {"relevant": False, "excerpts": []}

    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=2)


    print(f"[Finished Node] ground Ended In: {(datetime.now() - time).total_seconds()} seconds ({len(state.retrieved_articles)} articles in parallel)")
    return {"grounded_skeleton": skeleton_json}


# Node: apply the relevant law according to the grounder for each article: how does this article apply for the questions context, and then analyses the gaps in the question
# In case no relevant we go to the relevant fallback node
async def apply(state: State) -> dict:
    print("[Started Node] apply")
    time = datetime.now()

    skeleton = json.loads(state.grounded_skeleton)
    relevant = {k: v for k, v in skeleton.items() if v.get("relevant") is True}

    if not relevant:
        empty = json.dumps({
            "findings": [],
            "gaps": ["Aucun article pertinent trouvé pour cette question."]
        }, ensure_ascii=False)
        return {"apply_output": empty}

    apply_prompt = load_apply_prompt()

    # ── Step 1: parallel per-article findings ─────────────────
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

    # ── Step 2: global gap detection ──────────────────────────
    findings_text = json.dumps(findings, ensure_ascii=False, indent=2)

    gaps_response = await llm.ainvoke([
        SystemMessage(content=(
            "Analysez ces conclusions par rapport à la question. "
            "Une information est manquante UNIQUEMENT si elle changerait fondamentalement la réponse. "
            "Si la question est complètement répondue par les conclusions, retournez: [] "
            "Soyez conservateur — en cas de doute, retournez: [] "
            "Retournez uniquement une liste JSON de strings."
        )),
        HumanMessage(content=(
            f"Question: {state.input_text}\n\n"
            f"Conclusions:\n{findings_text}"
        ))
    ])

    try:
        raw_gaps = re.search(r'\[.*\]', gaps_response.content, re.DOTALL)
        gaps = json.loads(raw_gaps.group()) if raw_gaps else []
    except (json.JSONDecodeError, AttributeError):
        gaps = []

    # ── Output ────────────────────────────────────────────────
    apply_output = json.dumps(
        {"findings": findings, "gaps": gaps},
        ensure_ascii=False,
        indent=2
    )

    print(f"[Finished Node] Ended In: {(datetime.now() - time).total_seconds()} seconds ({len(relevant)} articles in parallel)")
    return {"apply_output": apply_output}


# Takes the answer given and compare the
async def critic_answer(state: State) -> dict:
    print("[Started Node] critic")
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
        print(f"[critic] claim extraction error: {e}")
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
            print(f"[critic] claim check error: {e}")
            return claim, True

    claim_results = await asyncio.gather(*[_check_claim(c) for c in claims]) if claims else []
    invented = [claim for claim, supported in claim_results if not supported]

    # Approved
    if not invented:
        print(f"[Finished Node] critic: APPROUVÉ  In: {(datetime.now() - time).total_seconds()} seconds")
        return {"critic_opinion": ""}

    # Build accumulated feedback
    feedback_lines = [f"- \"{claim}\" [IN ANSWER, NOT IN SKELETON]" for claim in invented]
    feedback = "\n".join(feedback_lines)

    print(f"[Finished Node] Critic: NON APPROUVÉ Ended In: {(datetime.now() - time).total_seconds()} seconds")

    if state.critic_opinion:
        accumulated = state.critic_opinion + f"\n\n[Round {state.retry_count + 1}]:\n{feedback}"
    else:
        accumulated = f"[Round 0]:\n{feedback}"

    return {"critic_opinion": accumulated}