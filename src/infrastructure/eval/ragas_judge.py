from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall, LLMContextPrecisionWithReference
from ragas.llms import llm_factory
from langgraph.graph.state import CompiledStateGraph

from src.domain.ports.judge import Judge
from src.domain.models.models import EvaluationResult


class RagasJudge(Judge):

    def __init__(self, evaluator_llm):
        self._metrics = [
            Faithfulness(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
        ]

    async def eval(self, dataset: list[dict], agent: CompiledStateGraph) -> EvaluationResult:
        faithfulness_scores = []
        factual_correctness_scores = []
        context_recall_scores = []
        context_precision_scores = []

        for item in dataset:
            print(f"[Evaluation] Evaluating question {item['id']}")
            result = await agent.ainvoke(
                {"input_text": item["question"]},
                config={"configurable": {"thread_id": f"eval-{item['id']}"}}
            )

            sample = SingleTurnSample(
                user_input=item["question"],
                reference=item["answer"],
                response=result["answer"],
                retrieved_contexts=[
                    f"{a.breadcrumb}\n{a.content}"
                    for a in result["retrieved_articles"]
                ],
            )

            for metric in self._metrics:
                print(f"[Evaluation] Evaluating {metric.name} for question {item['id']}")
                score = await metric.single_turn_ascore(sample)
                match metric.name:
                    case "faithfulness":
                        faithfulness_scores.append(score)
                    case "factual_correctness":
                        factual_correctness_scores.append(score)
                    case "context_recall":
                        context_recall_scores.append(score)
                    case "context_precision":
                        context_precision_scores.append(score)

        return EvaluationResult(
            faithfulness=faithfulness_scores,
            factual_correctness=factual_correctness_scores,
            context_recall=context_recall_scores,
            context_precision=context_precision_scores,
        )