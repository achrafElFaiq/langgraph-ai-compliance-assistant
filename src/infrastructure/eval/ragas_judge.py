import asyncio
import json
from pathlib import Path
from uuid import uuid4

from ragas import SingleTurnSample
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall, LLMContextPrecisionWithReference
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
        sem = asyncio.Semaphore(1)
        write_lock = asyncio.Lock()
        run_id = uuid4().hex
        results_path = Path("datasets/eval/results.json")
        results_path.write_text("[]")

        async def _write_result(item_id: str, scores: dict):
            # Serialize read-modify-write to avoid losing results under async concurrency.
            async with write_lock:
                existing = json.loads(results_path.read_text())
                existing.append({"id": item_id, **scores})
                results_path.write_text(json.dumps(existing, indent=2))

        async def _eval_item(item):
            async with sem:
                result = await agent.ainvoke(
                    {"input_text": item["question"]},
                    config={"configurable": {"thread_id": f"eval-{run_id}-{item['id']}"}}
                )

                if isinstance(result, dict):
                    answer = result.get("answer", "")
                    retrieved_articles = result.get("retrieved_articles") or []
                else:
                    answer = getattr(result, "answer", "")
                    retrieved_articles = getattr(result, "retrieved_articles", None) or []

                if not isinstance(answer, str):
                    answer = str(answer)

                retrieved_contexts = []
                for article in retrieved_articles:
                    if isinstance(article, dict):
                        breadcrumb = article.get("breadcrumb", "")
                        content = article.get("content", "")
                    else:
                        breadcrumb = getattr(article, "breadcrumb", "")
                        content = getattr(article, "content", "")

                    if breadcrumb or content:
                        retrieved_contexts.append(f"{breadcrumb}\n{content}".strip())

                sample = SingleTurnSample(
                    user_input=item["question"],
                    reference=item["answer"],
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                )
                print(f"[Scoring] {item['id']}")
                scores_list = await asyncio.gather(*[
                    metric.single_turn_ascore(sample)
                    for metric in self._metrics
                ])
                scores = {m.name: s for m, s in zip(self._metrics, scores_list)}
                await _write_result(item["id"], scores)
                return scores

        all_scores = await asyncio.gather(*[_eval_item(item) for item in dataset])

        faithfulness_scores, factual_correctness_scores, context_recall_scores, context_precision_scores = [], [], [], []
        for scores in all_scores:
            faithfulness = scores.get("faithfulness")
            factual = scores.get("factual_correctness")
            recall = scores.get("context_recall")
            precision = scores.get("context_precision", scores.get("llm_context_precision_with_reference"))

            if faithfulness is not None:
                faithfulness_scores.append(faithfulness)
            if factual is not None:
                factual_correctness_scores.append(factual)
            if recall is not None:
                context_recall_scores.append(recall)
            if precision is not None:
                context_precision_scores.append(precision)

        return EvaluationResult(
            faithfulness=faithfulness_scores,
            factual_correctness=factual_correctness_scores,
            context_recall=context_recall_scores,
            context_precision=context_precision_scores,
        )