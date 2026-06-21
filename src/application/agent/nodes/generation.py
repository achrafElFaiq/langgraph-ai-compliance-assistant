import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.application.agent.state import State
from src.config.init_llm import llm
from src.config.init_prompts import (
    load_answer_prompt,
    load_synthesis_prompt,
)

def direct_answer(state: State) -> dict:
    print("[Current Node] direct_answer")

    response = llm.invoke([
        SystemMessage(content=(
            "Vous êtes un assistant de conformité réglementaire européenne. "
            "Répondez de manière naturelle et concise."
        )),
        *state.messages,
        HumanMessage(content=state.input_text)
    ])

    print("[Finished Node] direct_answer")

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state.input_text),
            AIMessage(content=response.content)
        ]
    }


# Node: takes all the finding and formulates One correct answer
def answer(state: State) -> dict:
    print("[Started Node] answer")
    time = datetime.now()

    apply_data = json.loads(state.apply_output)
    findings = apply_data.get("findings", [])
    gaps = apply_data.get("gaps", [])

    critic_section = ""
    retry_count = state.retry_count
    if state.critic_opinion:
        critic_section = (
            f"\n\nCette réponse a déjà été vérifiée. "
            f"Corrigez les points suivants sans exception:\n{state.critic_opinion}"
        )
        retry_count += 1

    findings_text = json.dumps(findings, ensure_ascii=False, indent=2)
    gaps_text = json.dumps(gaps, ensure_ascii=False, indent=2)

    response = llm.invoke([
        SystemMessage(content=load_answer_prompt()),
        *state.messages,
        HumanMessage(content=(
            f"Question: {state.input_text}\n\n"
            f"Conclusions:\n{findings_text}\n\n"
            f"Informations manquantes:\n{gaps_text}"
            f"{critic_section}"
        ))
    ])

    print(f"[Finished Node] answer In: {(datetime.now() - time).total_seconds()} seconds")
    return {
        "answer": response.content,
        "retry_count": retry_count,
        "messages": [
            HumanMessage(content=state.input_text),
            AIMessage(content=response.content)
        ]
    }


def synthesize(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=load_synthesis_prompt()),
        *state.messages,
        HumanMessage(content="Generate the compliance report based on our conversation.")
    ])
    return {"final_report": response.content}