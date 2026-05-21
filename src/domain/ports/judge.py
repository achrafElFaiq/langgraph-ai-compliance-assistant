from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph
from src.domain.models.models import EvaluationResult


class Judge(ABC):


    @abstractmethod
    async def eval(self, dataset: list[dict], agent: CompiledStateGraph) -> EvaluationResult:
        pass






