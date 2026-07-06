import asyncio
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from uuid import uuid4

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from src.config.init_langfuse import langfuse_handler
from src.domain.models.models import EvaluationResult
from src.domain.ports.judge import Judge
from src.infrastructure.eval.utils import print_results


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






class DeepEvalJudge(Judge):

    def __init__(self, model: str = "gpt-4o-mini"):
        self._metrics = [
            FaithfulnessMetric(threshold=0.5, model=model, async_mode=True),
            ContextualRecallMetric(threshold=0.5, model=model, async_mode=True),
            ContextualPrecisionMetric(threshold=0.5, model=model, async_mode=True),
            AnswerRelevancyMetric(threshold=0.5, model=model, async_mode=True),
        ]

    async def eval(self, dataset: list[dict], agent: CompiledStateGraph) -> EvaluationResult:
        sem = asyncio.Semaphore(5)
        write_lock = asyncio.Lock()
        run_id = uuid4().hex
        results_path = Path("datasets/agent-eval/results_deepeval.json")
        results_path.write_text("[]")

        async def _write_result(item_id: str, scores: dict):
            async with write_lock:
                existing = json.loads(results_path.read_text())
                existing.append({"id": item_id, **scores})
                results_path.write_text(json.dumps(existing, indent=2))

        async def _eval_item(item):
            async with sem:
                # Stream in debug+values mode: debug events give per-node timing,
                # the final "values" chunk is the final state (like ainvoke's return).
                node_starts: dict = {}
                node_latencies: dict[str, list[float]] = {}
                result: dict = {}

                start = time.perf_counter()
                async for mode, chunk in agent.astream(
                    {"input_text": item["question"]},
                    config={
                        "callbacks": [langfuse_handler],
                        "configurable": {"thread_id": f"eval-deepeval-{run_id}-{item['id']}"}
                    },
                    stream_mode=["debug", "values"],
                ):
                    if mode == "values":
                        result = chunk
                        continue
                    payload = chunk.get("payload", {})
                    if chunk.get("type") == "task":
                        node_starts[payload.get("id")] = time.perf_counter()
                    elif chunk.get("type") == "task_result":
                        t0 = node_starts.pop(payload.get("id"), None)
                        if t0 is not None:
                            name = payload.get("name", "")
                            node_latencies.setdefault(name, []).append(time.perf_counter() - t0)
                end_to_end = time.perf_counter() - start

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
                    breadcrumb = article.get("breadcrumb", "") if isinstance(article, dict) else getattr(article, "breadcrumb", "")
                    content = article.get("content", "") if isinstance(article, dict) else getattr(article, "content", "")
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
                scores["latency"] = end_to_end
                scores["node_latencies"] = node_latencies
                await _write_result(item["id"], scores)
                return scores

        await asyncio.gather(*[_eval_item(item) for item in dataset])

        results_data =json.loads(results_path.read_text())
        print_results(results_data)

        faithfulness = [e.get("FaithfulnessMetric", 0)       for e in results_data]
        recall       = [e.get("ContextualRecallMetric", 0)    for e in results_data]
        precision    = [e.get("ContextualPrecisionMetric", 0) for e in results_data]
        relevancy    = [e.get("AnswerRelevancyMetric", 0)     for e in results_data]

        end_to_end_latency = [e.get("latency", 0.0) for e in results_data]
        retry_counts       = [e.get("retry_count", 0) for e in results_data]

        # Flatten per-question node timings into one list of durations per node.
        node_latencies: dict[str, list[float]] = {}
        for e in results_data:
            for node, durs in (e.get("node_latencies") or {}).items():
                node_latencies.setdefault(node, []).extend(durs)

        return EvaluationResult(
            faithfulness=faithfulness,
            factual_correctness=relevancy,
            context_recall=recall,
            context_precision=precision,
            end_to_end_latency=end_to_end_latency,
            node_latencies=node_latencies,
            retry_counts=retry_counts,
        )