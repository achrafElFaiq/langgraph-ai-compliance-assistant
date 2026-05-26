import asyncio
import json
import logging
import os
import re
from datetime import datetime

from src.config.init_llm import llm, grounder_llm
from src.config.init_llm import critic_llm
from src.config.init_embedder import embedder
from src.config.init_store import store
from src.config.init_prompts import (
    load_question_generation_prompt,
    load_needs_research_prompt,
    load_answer_prompt,
    load_critic_prompt,
    load_synthesis_prompt, load_ground_prompt,
)

from src.application.agent.state import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def write_to_file(content: str, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def generate_questions(state: State) -> dict:
    print("[Current Node] generate questions")
    time = datetime.now()
    user_content = state.input_text

    write_to_file(f" {load_question_generation_prompt()}\n\n{user_content}", f"run/prmpottogeneratequestion.md")
    response = llm.invoke([
        SystemMessage(content=load_question_generation_prompt()),
        HumanMessage(content=user_content),
    ])

    write_to_file(response.content, f"run/generatedquestion.md")
    print("Generation Ended In : ", (datetime.now() - time).total_seconds(), " seconds")
    return {"retrieval_query": response.content}


def needs_research(state: State) -> dict:
    print("[Current Node] Needs Research")
    if not state.messages or state.critic_opinion:
        print("Needs Research: True")
        return {"needs_research": True}

    response = llm.invoke([
        SystemMessage(content=load_needs_research_prompt()),
        HumanMessage(content=state.retrieval_query)
    ])
    needs_retrieval = response.content.strip().upper() == "RETRIEVE"
    print("Needs Research: ", needs_retrieval)
    return {"needs_research": needs_retrieval}

async def retrieve_articles(state: State) -> dict:
    print("[Current Node] retrieve articles")
    time = datetime.now()
    query = state.retrieval_query
    embedding = await embedder.embed_query(query)
    articles = await store.retrieve(embedding=embedding, query=query)

    skeleton = {
        a.breadcrumb: {"relevant": None, "excerpt_basis": "", "how_it_applies": ""}
        for a in articles
    }

    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=2)
    print(f"Retrieved {len(articles)} articles")
    print("Research Ended In : ", (datetime.now() - time).total_seconds(), " seconds")
    return {
        "retrieved_articles": articles,
        "answer": skeleton_json
    }


async def ground(state: State) -> dict:
    print("[Current Node] ground")
    os.makedirs("run", exist_ok=True)
    os.makedirs("datasets/skeletons", exist_ok=True)
    time = datetime.now()

    ground_prompt = load_ground_prompt()

    async def _ground_article(article):
        response = await grounder_llm.ainvoke([
            SystemMessage(content=ground_prompt),
            HumanMessage(content=f"Query: {state.retrieval_query}\n\nArticle ({article.breadcrumb}):\n{article.content}")
        ])
        raw = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not raw:
            return article.breadcrumb, {"relevant": False, "excerpt_basis": "", "how_it_applies": "Non applicable."}
        try:
            parsed = json.loads(raw.group())
        except json.JSONDecodeError:
            parsed = {"relevant": False, "excerpt_basis": "", "how_it_applies": "Non applicable."}
        return article.breadcrumb, parsed

    results = await asyncio.gather(*[_ground_article(a) for a in state.retrieved_articles])

    skeleton = {breadcrumb: slots for breadcrumb, slots in results}
    skeleton_json = json.dumps(skeleton, ensure_ascii=False, indent=2)

    write_to_file(skeleton_json, "run/grounderresponse.md")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = state.retrieval_query[:30].replace(" ", "_")
    with open(f"datasets/skeletons/{timestamp}_{query_slug}.json", "w", encoding="utf-8") as f:
        f.write(skeleton_json)

    print(f"Grounding Ended In : {(datetime.now() - time).total_seconds()} seconds ({len(state.retrieved_articles)} articles in parallel)")
    return {"answer": skeleton_json}

def answer(state: State) -> dict:
    print("[Current Node] answer")
    time = datetime.now()

    raw = state.answer
    if "---" in raw:
        raw = raw.split("---", 1)[1].strip()

    skeleton = json.loads(raw)
    relevant = {k: v for k, v in skeleton.items() if v.get("relevant") is True}
    relevant_json = json.dumps(relevant, ensure_ascii=False, indent=2)

    critic_section = ""
    retry_count = state.retry_count
    if state.critic_opinion:
        critic_section = (
            f"\n\nThis answer has been reviewed. Below is the full accumulated critic feedback across all rounds "
            f"— every point must be addressed in your response:\n{state.critic_opinion}\n"
        )
        retry_count += 1

    write_to_file(f"System: {load_answer_prompt()} \n\n  User input: {state.input_text}\n\nRetrieval query: {state.retrieval_query}{critic_section}\n\nSkeleton to answer from:\n{relevant_json}", f"run/promptforganswer.md")
    response = llm.invoke([
        SystemMessage(content=load_answer_prompt()),
        *state.messages,
        HumanMessage(content=f"User input: {state.input_text}\n\nRetrieval query: {state.retrieval_query}{critic_section}\n\nSkeleton to answer from:\n{relevant_json}")
    ])

    final_answer = f"{response.content}\n\n---\n{relevant_json}"
    print("Answering Ended In : ", (datetime.now() - time).total_seconds(), " seconds")
    return {
        "answer": final_answer,
        "retry_count": retry_count,
        "messages": [
            HumanMessage(content=state.input_text),
            AIMessage(content=final_answer)
        ]
    }


def critic_answer(state: State) -> dict:
    print("[Current Node] critic ")
    time = datetime.now()
    articles_text = "\n\n".join(
        f"{a.breadcrumb}:\n{a.content}"
        for a in state.retrieved_articles
    )

    if state.critic_opinion:
        user_content = (
            f"Previous feedback you gave:\n{state.critic_opinion}\n\n"
            f"New answer to verify:\n{state.answer}\n\n"
            f"Articles:\n{articles_text}"
        )
    else:
        user_content = f"Answer: {state.answer}\n\nArticles:\n{articles_text}"

    write_to_file(user_content, "run/promptforcritic.md")
    response = critic_llm.invoke([
        SystemMessage(content=load_critic_prompt()),
        HumanMessage(content=user_content)
    ])

    feedback = response.content.strip()
    write_to_file(feedback, "run/criticfeedback.md")
    print("Critic Ended In : ", (datetime.now() - time).total_seconds(), " seconds")
    print("Critic Feedback: ", feedback)
    if feedback.upper() == "APPROVED":
        return {"critic_opinion": ""}

    if state.critic_opinion:
        accumulated = state.critic_opinion + f"\n\n[Round {state.retry_count + 1} remaining issues]:\n{feedback}"
    else:
        accumulated = feedback

    return {"critic_opinion": accumulated}

def synthesize(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=load_synthesis_prompt()),
        *state.messages,
        HumanMessage(content="Generate the compliance report based on our conversation.")
    ])
    return {"final_report": response.content}