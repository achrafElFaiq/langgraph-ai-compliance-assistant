import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.application.agent.state import State
from src.application.agent.nodes.intent import classify_intent
from src.application.agent.nodes.retrieval import classify, retrieve_articles, retrieve_fallback
from src.application.agent.nodes.reasoning import ground, apply, critic_answer
from src.application.agent.nodes.generation import direct_answer, answer, synthesize

checkpointer = MemorySaver()


def route_intent(state: State) -> str:
    return state.route


def route_after_ground(state: State) -> str:
    skeleton = json.loads(state.grounded_skeleton)
    relevant = [v for v in skeleton.values() if v.get("relevant") is True]
    if not relevant and not state.fallback_attempted:
        return "retrieve_fallback"
    return "apply"


def route_critic(state: State) -> str:
    if state.critic_opinion == "" or state.retry_count >= 2:
        return "END"
    return "answer"


graph = StateGraph(State)

graph.add_node("classify_intent",   classify_intent)
graph.add_node("direct_answer",     direct_answer)
graph.add_node("classify",          classify)
graph.add_node("retrieve_articles", retrieve_articles)
graph.add_node("retrieve_fallback", retrieve_fallback)
graph.add_node("ground",            ground)
graph.add_node("apply",             apply)
graph.add_node("answer",            answer)
graph.add_node("critic_answer",     critic_answer)
graph.add_node("synthesize",        synthesize)

graph.add_edge(START, "classify_intent")
graph.add_conditional_edges("classify_intent", route_intent, {
    "synthesis": "synthesize",
    "chitchat":  "direct_answer",
    "followup":  "answer",
    "research":  "classify",
})

graph.add_edge("direct_answer",     END)
graph.add_edge("classify",          "retrieve_articles")
graph.add_edge("retrieve_articles", "ground")

graph.add_conditional_edges("ground", route_after_ground, {
    "retrieve_fallback": "retrieve_fallback",
    "apply":             "apply"
})

graph.add_edge("retrieve_fallback", "ground")
graph.add_edge("apply",             "answer")
graph.add_edge("answer",            "critic_answer")

graph.add_conditional_edges("critic_answer", route_critic, {
    "answer": "answer",
    "END":    END
})

graph.add_edge("synthesize", END)

compiled_graph = graph.compile(checkpointer=checkpointer)