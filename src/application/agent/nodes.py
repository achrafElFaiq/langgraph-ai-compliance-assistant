from src.config.init_llm import llm
from src.config.init_embedder import embedder
from src.config.init_store import store
from src.config.init_prompts import (
    load_question_generation_prompt,
    load_needs_research_prompt,
    load_answer_prompt,
    load_critic_prompt,
    load_synthesis_prompt,
)

from src.application.agent.state import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def generate_questions(state: State) -> dict:
    print("Generating questions...")
    response = llm.invoke(
        [
            SystemMessage(content=load_question_generation_prompt()),
            HumanMessage(content=state["input_text"]),
        ]
    )
    print("Generated questions:", response.content)
    return {"retrieval_query": response.content}


def needs_research(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=load_needs_research_prompt()),
        HumanMessage(content=state["retrieval_query"])
    ])
    needs_retrieval = response.content.strip().upper() == "RETRIEVE"
    print("Needs retrieval:", response.content)
    return {"needs_research": needs_retrieval}


async def retrieve_articles(state: State) -> dict:
    query = state["retrieval_query"]
    embedding = await embedder.embed_query(query)
    articles = await store.retrieve(embedding=embedding, query=query)
    print("Retrieved articles:", len(articles))
    return {"retrieved_articles": articles}


def answer(state: State) -> dict:
    articles_text = "\n\n".join([
        f"{a.breadcrumb}:\n{a.content}"
        for a in state["retrieved_articles"]
    ])

    response = llm.invoke([
        SystemMessage(content=load_answer_prompt()),
        *state["messages"],
        HumanMessage(content=f"Query: {state['retrieval_query']}\n\nArticles:\n{articles_text}")
    ])

    print("Answer:", response.content)
    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["input_text"]),
            AIMessage(content=response.content)
        ]
    }


def critic_answer(state: State) -> dict:
    articles_text = "\n\n".join([
        f"{a.breadcrumb}:\n{a.content}"
        for a in state["retrieved_articles"]
    ])

    response = llm.invoke([
        SystemMessage(content=load_critic_prompt()),
        HumanMessage(content=f"Answer: {state['answer']}\n\nArticles:\n{articles_text}")
    ])

    feedback = response.content.strip()
    print("Critic answer:", response.content)
    if feedback.upper() == "APPROVED":
        return {"critic_feedback": ""}
    return {"critic_feedback": feedback}


def synthesize(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=load_synthesis_prompt()),
        *state["messages"],
        HumanMessage(content="Generate the compliance report based on our conversation.")
    ])
    return {"final_report": response.content}