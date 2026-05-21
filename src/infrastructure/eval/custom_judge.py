from langgraph.graph.state import CompiledStateGraph

from domain.models.models import EvaluationResult
from domain.ports.judge import Judge


class CustomJudge(Judge):


    async def eval(self, dataset: list[dict], agent: CompiledStateGraph) -> EvaluationResult:
        pass