from src.config.llm import llm
from src.config.embedder import embedder
from src.config.store import store




from src.core.agent.state import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage



GENERATE_QUERY_PROMPT = """You are a compliance expert specializing in French financial regulations (MiCA, DORA, GDPR, AMF, ACPR).

Given the user input, produce a single dense paragraph that captures all compliance-relevant questions and context.
This paragraph will be used to search a regulatory database — be specific about regulations, obligations, and legal concepts.
Do not answer the question. Only produce the search query paragraph."""


NEEDS_RESEARCH_PROMPT = """You are a compliance expert. Given the following retrieval query and conversation history, decide if you need to retrieve regulation articles from the database to answer it.

Reply with only one word:
- RETRIEVE if you need to look up specific regulation articles
- DIRECT if you can answer from the conversation history or general knowledge"""

ANSWER_PROMPT = """You are a compliance expert specializing in French financial regulations (MiCA, DORA, GDPR, AMF, ACPR).

Answer the user's compliance question based on the provided regulation articles.
- Ground every claim in a specific article
- Cite the article number and regulation name for each point
- If the articles don't contain enough information to answer, say so clearly
- Be precise and avoid speculation"""


CRITIC_PROMPT = """You are a compliance review expert. Evaluate the following answer against the provided regulation articles.

Check:
- Is every claim grounded in a specific article?
- Are citations accurate?
- Is anything missing or misrepresented?

If the answer is satisfactory reply with only: APPROVED
If not, reply with a short explanation of what is wrong or missing."""

SYNTHESIZE_PROMPT = """You are a compliance expert. Based on the full conversation history, produce a structured compliance report.

The report should include:
- Executive summary
- Key compliance obligations identified
- Applicable regulations (MiCA, DORA, GDPR, AMF, ACPR)
- Risks and gaps identified
- Recommended actions

Be precise, cite specific articles where relevant."""

def generate_questions(state: State) -> dict:
    print("Generating questions...")
    response = llm.invoke(
        [
            SystemMessage(content=GENERATE_QUERY_PROMPT),
            HumanMessage(content=state["input_text"]),
        ]
    )
    print("Generated questions:", response.content)
    return {"retrieval_query": response.content}


def needs_research(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=NEEDS_RESEARCH_PROMPT),
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
    return {"retrieved_articles": articles }


def answer(state: State) -> dict:
    articles_text = "\n\n".join([
        f"{a.breadcrumb}:\n{a.content}"
        for a in state["retrieved_articles"]
    ])

    response = llm.invoke([
        SystemMessage(content=ANSWER_PROMPT),
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
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=f"Answer: {state['answer']}\n\nArticles:\n{articles_text}")
    ])

    feedback = response.content.strip()
    print("Critic answer:", response.content)
    if feedback.upper() == "APPROVED":
        return {"critic_feedback": ""}
    return {"critic_feedback": feedback}

def synthesize(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=SYNTHESIZE_PROMPT),
        *state["messages"],
        HumanMessage(content="Generate the compliance report based on our conversation.")
    ])
    return {"final_report": response.content}