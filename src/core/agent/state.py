from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from src.domain.models.models import Article


class State(TypedDict):
    input_text: str
    retrieval_query : str
    needs_research: bool
    retrieved_articles: Annotated[list[Article], "List of articles retrieved from the database based on the retrieval query."]
    answer: str
    critic_feedback : str
    retry_count : int
    final_report : str
    messages : Annotated[list,add_messages]

