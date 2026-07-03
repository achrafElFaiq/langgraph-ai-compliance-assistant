import logging
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import subprocess

import mlflow

from src.infrastructure.eval.deepeval_judge import DeepEvalJudge, OpenRouterDeepEvalLLM
from src.application.agent.graph import compiled_graph
from src.config.init_store import store
from src.config.init_llm import llm, grounder_llm, critic_llm

logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 0.7


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _load_classifier_thresholds() -> dict:
    import joblib
    mlb = joblib.load("models/mlb.joblib")
    thresholds = joblib.load("models/thresholds.joblib")
    return {reg: float(thresholds[i]) for i, reg in enumerate(mlb.classes_)}


async def run_evaluation_pipeline(dataset_path: str = "datasets/agent-eval/dataset.json"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    logger.info("eval | started n_questions=%d", len(dataset))

    await store.connect()
    try:
        model = OpenRouterDeepEvalLLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=8192
        )
        judge = DeepEvalJudge(model=model)

        mlflow.set_experiment("compliance-assistant")

        with mlflow.start_run():
            # ── Parameters ────────────────────────────────────────────
            mlflow.log_param("git_commit", _git_commit())
            mlflow.log_param("llm_model", llm.model_name)
            mlflow.log_param("grounder_model", grounder_llm.model_name)
            mlflow.log_param("critic_model", critic_llm.model_name)
            mlflow.log_param("top_k_formula", "5 * len(regulations)")
            mlflow.log_param("fallback_top_k", 5)
            mlflow.log_param("retry_cap", 2)
            mlflow.log_param("eval_judge_model", "openai/gpt-4o-mini")
            mlflow.log_param("n_questions", len(dataset))

            # Classifier thresholds
            thresholds = _load_classifier_thresholds()
            for reg, thresh in thresholds.items():
                mlflow.log_param(f"threshold_{reg}", thresh)

            # ── Prompt artifacts ──────────────────────────────────────
            for prompt_file in Path("configs/prompts").glob("*.yaml"):
                mlflow.log_artifact(str(prompt_file), artifact_path="prompts")

            # Also log regulations config
            mlflow.log_artifact("configs/regulations.yaml", artifact_path="prompts")

            # ── Graph structure ───────────────────────────────────────
            with tempfile.TemporaryDirectory() as tmp:
                graph_png_path = os.path.join(tmp, "graph.png")
                png_bytes = compiled_graph.get_graph().draw_mermaid_png()
                with open(graph_png_path, "wb") as f:
                    f.write(png_bytes)
                mlflow.log_artifact(graph_png_path, artifact_path="graph")

            # ── Run eval ──────────────────────────────────────────────
            result = await judge.eval(dataset=dataset, agent=compiled_graph)

            # ── Per-question logging ───────────────────────────────────
            for i in range(len(result.faithfulness)):
                q_num = i + 1
                faith = result.faithfulness[i]
                recall = result.context_recall[i]
                logger.info(
                    "eval | q=%d faithfulness=%.3f factual=%.3f ctx_recall=%.3f ctx_precision=%.3f",
                    q_num, faith, result.factual_correctness[i], recall, result.context_precision[i],
                )
                if faith < _WARN_THRESHOLD or recall < _WARN_THRESHOLD:
                    logger.warning(
                        "eval | q=%d below threshold faithfulness=%.3f ctx_recall=%.3f threshold=%.1f",
                        q_num, faith, recall, _WARN_THRESHOLD,
                    )
                logger.debug(
                    "eval | q=%d full_metrics=%s",
                    q_num,
                    {
                        "faithfulness": faith,
                        "factual_correctness": result.factual_correctness[i],
                        "context_recall": recall,
                        "context_precision": result.context_precision[i],
                        "question": dataset[i].get("question", dataset[i].get("user_input", "")),
                    },
                )

            # ── Aggregate metrics ─────────────────────────────────────
            def mean(lst):
                return sum(lst) / len(lst) if lst else 0.0

            mlflow.log_metric("faithfulness", mean(result.faithfulness))
            mlflow.log_metric("factual", mean(result.factual_correctness))
            mlflow.log_metric("ctx_recall", mean(result.context_recall))
            mlflow.log_metric("ctx_precision", mean(result.context_precision))

            # ── Latency & retry metrics ───────────────────────────────
            mlflow.log_metric("mean_latency_e2e_s", mean(result.end_to_end_latency))
            mlflow.log_metric("mean_retry_count", mean(result.retry_counts))
            for node, durations in result.node_latencies.items():
                mlflow.log_metric(f"mean_latency_node_{node}_s", mean(durations))

            logger.info(
                "eval | latency_e2e=%.2fs mean_retries=%.2f",
                mean(result.end_to_end_latency), mean(result.retry_counts),
            )

            # ── Per-question metrics ───────────────────────────────────
            for i in range(len(result.faithfulness)):
                step = i + 1
                mlflow.log_metric("faithfulness_per_q", result.faithfulness[i], step=step)
                mlflow.log_metric("factual_per_q", result.factual_correctness[i], step=step)
                mlflow.log_metric("ctx_recall_per_q", result.context_recall[i], step=step)
                mlflow.log_metric("ctx_precision_per_q", result.context_precision[i], step=step)

            # ── Per-question JSON artifact ─────────────────────────────
            per_q = {
                f"r{i+1}": {
                    "question": dataset[i].get("question", dataset[i].get("user_input", "")),
                    "faithfulness": result.faithfulness[i],
                    "factual": result.factual_correctness[i],
                    "ctx_recall": result.context_recall[i],
                    "ctx_precision": result.context_precision[i],
                }
                for i in range(len(result.faithfulness))
            }
            with tempfile.TemporaryDirectory() as results_tmp:
                per_q_path = os.path.join(results_tmp, "per_question.json")
                Path(per_q_path).write_text(json.dumps(per_q, indent=2, ensure_ascii=False))
                mlflow.log_artifact(per_q_path, artifact_path="results")

            logger.info(
                "eval | finished faith=%.3f factual=%.3f recall=%.3f precision=%.3f",
                mean(result.faithfulness), mean(result.factual_correctness),
                mean(result.context_recall), mean(result.context_precision),
            )
    finally:
        await store.close()


async def main():
    await run_evaluation_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
