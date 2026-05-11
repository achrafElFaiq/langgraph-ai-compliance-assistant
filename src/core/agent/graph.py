from langgraph.graph import StateGraph, START, END
from src.core.agent.state import State
from langgraph.checkpoint.memory import MemorySaver

from src.core.agent.nodes import (
    generate_questions,
    needs_research,
    retrieve_articles,
    answer,
    critic_answer,
    synthesize
)


checkpointer = MemorySaver()

def needs_report(state: State) -> str:
    if state["input_text"] == "Generate synthesis":
        return "synthesize"
    return "generate_query"

def route_research(state: State) -> str:
    if state["needs_research"]:
        return "retrieve_articles"
    return "answer"

def route_critic(state: State) -> str:
    if state["critic_feedback"]:
        return "generate_query"
    return "END"

graph = StateGraph(State)


graph.add_node("generate_query", generate_questions)
graph.add_node("needs_research", needs_research)
graph.add_node("retrieve_articles", retrieve_articles)
graph.add_node("answer", answer)
graph.add_node("critic_answer", critic_answer)
graph.add_node("synthesize", synthesize)


graph.add_conditional_edges(START, needs_report, {
    "synthesize": "synthesize",
    "generate_query": "generate_query"
})

graph.add_edge("generate_query", "needs_research")

graph.add_conditional_edges("needs_research", route_research, {
    "retrieve_articles": "retrieve_articles",
    "answer": "answer"
})

graph.add_edge("retrieve_articles", "answer")
graph.add_edge("answer", "critic_answer")


graph.add_conditional_edges("critic_answer", route_critic, {
    "generate_query": "generate_query",
    "END": END
})

graph.add_edge("synthesize", END)
compiled_graph = graph.compile(checkpointer=checkpointer)