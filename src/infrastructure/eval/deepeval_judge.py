from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from typing import Optional, Tuple, Union
from deepeval.models.base_model import DeepEvalBaseLLM

from src.config.init_langfuse import langfuse_handler


class OpenRouterDeepEvalLLM(DeepEvalBaseLLM):

    def __init__(self, model: str, api_key: str, max_tokens: int = 8192):
        self.model_name = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
        )

    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: Optional[type[BaseModel]] = None) -> Tuple[Union[str, BaseModel], float]:
        if schema:
            completion = self._client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.parsed, 0.0
        else:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content, 0.0

    async def a_generate(self, prompt: str, schema: Optional[type[BaseModel]] = None) -> Tuple[Union[str, BaseModel], float]:
        if schema:
            completion = await self._async_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.parsed, 0.0
        else:
            completion = await self._async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content, 0.0

    def supports_structured_outputs(self) -> bool:
        return True



import asyncio
import json
from pathlib import Path
from uuid import uuid4

from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from langgraph.graph.state import CompiledStateGraph

from src.domain.ports.judge import Judge
from src.domain.models.models import EvaluationResult


class DeepEvalJudge(Judge):

    def __init__(self, model: str = "gpt-4o-mini"):
        self._metrics = [
            FaithfulnessMetric(threshold=0.5, model=model, async_mode=True),
            ContextualRecallMetric(threshold=0.5, model=model, async_mode=True),
            ContextualPrecisionMetric(threshold=0.5, model=model, async_mode=True),
            AnswerRelevancyMetric(threshold=0.5, model=model, async_mode=True),
        ]

    async def eval(self, dataset: list[dict], agent: CompiledStateGraph) -> EvaluationResult:
        sem = asyncio.Semaphore(1)
        write_lock = asyncio.Lock()
        run_id = uuid4().hex
        results_path = Path("datasets/eval/results_deepeval.json")
        results_path.write_text("[]")

        async def _write_result(item_id: str, scores: dict):
            async with write_lock:
                existing = json.loads(results_path.read_text())
                existing.append({"id": item_id, **scores})
                results_path.write_text(json.dumps(existing, indent=2))

        async def _eval_item(item):
            async with sem:
                result = await agent.ainvoke(
                    {"input_text": item["question"]},
                    config={
                        "callbacks": [langfuse_handler],
                        "configurable": {"thread_id": f"eval-deepeval-{run_id}-{item['id']}"}
                    }
                )

                if isinstance(result, dict):
                    answer = result.get("answer", "")
                    retrieved_articles = result.get("retrieved_articles") or []
                    retry_count = result.get("retry_count", 0)
                else:
                    answer = getattr(result, "answer", "")
                    retrieved_articles = getattr(result, "retrieved_articles", None) or []
                    retry_count = getattr(result, "retry_count", 0)

                if not isinstance(answer, str):
                    answer = str(answer)

                if "---" in answer:
                    answer = answer.split("---", 1)[0].strip()

                retrieval_context = []
                for article in retrieved_articles:
                    breadcrumb = article.get("breadcrumb", "") if isinstance(article, dict) else getattr(article,
                                                                                                         "breadcrumb",
                                                                                                         "")
                    content = article.get("content", "") if isinstance(article, dict) else getattr(article, "content",
                                                                                                   "")
                    if breadcrumb or content:
                        retrieval_context.append(f"{breadcrumb}\n{content}".strip())

                test_case = LLMTestCase(
                    input=item["question"],
                    actual_output=answer,
                    expected_output=item["answer"],
                    retrieval_context=retrieval_context,
                    context=retrieval_context,
                )

                print(f"[Scoring] {item['id']}")
                scores_list = await asyncio.gather(*[
                    metric.a_measure(test_case, _show_indicator=False)
                    for metric in self._metrics
                ])
                scores = {m.__class__.__name__: s for m, s in zip(self._metrics, scores_list)}
                scores["retry_count"] = retry_count
                await _write_result(item["id"], scores)
                return scores

        all_scores = await asyncio.gather(*[_eval_item(item) for item in dataset])

        faithfulness, recall, precision, relevancy, hallucination, retries = [], [], [], [], [], []
        for scores in all_scores:
            faithfulness.append(scores.get("FaithfulnessMetric", 0))
            recall.append(scores.get("ContextualRecallMetric", 0))
            precision.append(scores.get("ContextualPrecisionMetric", 0))
            relevancy.append(scores.get("AnswerRelevancyMetric", 0))
            hallucination.append(scores.get("HallucinationMetric", 0))
            retries.append(scores.get("retry_count", 0))

        print("\n=== DeepEval Results ===")
        print(f"Faithfulness:     {sum(faithfulness) / len(faithfulness):.3f}")
        print(f"Answer Relevancy: {sum(relevancy) / len(relevancy):.3f}")
        print(f"Ctx Recall:       {sum(recall) / len(recall):.3f}")
        print(f"Ctx Precision:    {sum(precision) / len(precision):.3f}")
        print(f"Hallucination:    {sum(hallucination) / len(hallucination):.3f}")

        print("\n=== Retry Stats ===")
        print(f"Avg retries:      {sum(retries) / len(retries):.2f}")
        print(f"Max retries:      {max(retries)}")
        print(f"Hit cap (3):      {sum(1 for r in retries if r >= 3)} / {len(retries)} questions")
        print("\n  Per question:")
        for scores in sorted(all_scores, key=lambda x: x.get("retry_count", 0), reverse=True):
            item_id = next(
                item["id"] for item in dataset
                if f"eval-deepeval-{run_id}-{item['id']}" or True
            )

        results_data = json.loads(results_path.read_text())
        for entry in sorted(results_data, key=lambda x: x.get("retry_count", 0), reverse=True):
            r = entry.get("retry_count", 0)
            bar = "█" * r + "░" * (3 - min(r, 3))
            flag = " ← hit cap" if r >= 3 else ""
            print(f"    {entry['id']:<6} {bar} {r} retries{flag}")

        return EvaluationResult(
            faithfulness=faithfulness,
            factual_correctness=relevancy,
            context_recall=recall,
            context_precision=precision,
        )