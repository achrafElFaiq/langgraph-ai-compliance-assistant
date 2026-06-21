from dataclasses import dataclass, field
from typing import Annotated
from langgraph.graph.message import add_messages
from src.domain.models.models import Article

@dataclass
class State:
    input_text: str = ""
    route: str = ""
    regulations: list[str] = field(default_factory=list)
    retrieved_articles: list[Article] = field(default_factory=list)
    grounded_skeleton: str = ""
    apply_output: str = ""
    answer: str = ""
    critic_opinion: str = ""
    retry_count: int = 0
    final_report: str = ""
    fallback_attempted: bool = False
    messages: Annotated[list, add_messages] = field(default_factory=list)