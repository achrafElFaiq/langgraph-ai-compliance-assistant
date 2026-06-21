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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _load_classifier_thresholds() -> dict:
    import joblib
    mlb = joblib.load("datasets/classifier/model/mlb.joblib")
    thresholds = joblib.load("datasets/classifier/model/thresholds.joblib")
    return {reg: float(thresholds[i]) for i, reg in enumerate(mlb.classes_)}


async def run_evaluation_pipeline(dataset_path: str = "datasets/eval/dataset.json"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

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
            os.makedirs("run", exist_ok=True)
            graph_png_path = "run/graph.png"
            png_bytes = compiled_graph.get_graph().draw_mermaid_png()
            with open(graph_png_path, "wb") as f:
                f.write(png_bytes)
            mlflow.log_artifact(graph_png_path, artifact_path="graph")




            # ── Run eval ──────────────────────────────────────────────
            result = await judge.eval(dataset=dataset, agent=compiled_graph)

            # ── Aggregate metrics ─────────────────────────────────────
            def mean(lst):
                return sum(lst) / len(lst) if lst else 0.0

            mlflow.log_metric("faithfulness", mean(result.faithfulness))
            mlflow.log_metric("factual", mean(result.factual_correctness))
            mlflow.log_metric("ctx_recall", mean(result.context_recall))
            mlflow.log_metric("ctx_precision", mean(result.context_precision))

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
            per_q_path = "run/per_question.json"
            Path(per_q_path).write_text(json.dumps(per_q, indent=2, ensure_ascii=False))
            mlflow.log_artifact(per_q_path, artifact_path="results")

            # ── Print summary ─────────────────────────────────────────
            print(f"\nFaithfulness:  {result.faithfulness}")
            print(f"Factual:       {result.factual_correctness}")
            print(f"Ctx Recall:    {result.context_recall}")
            print(f"Ctx Precision: {result.context_precision}")
            print(
                f"\nMeans → faith={mean(result.faithfulness):.3f} | factual={mean(result.factual_correctness):.3f} | recall={mean(result.context_recall):.3f} | precision={mean(result.context_precision):.3f}"
            )
    finally:
        await store.close()


async def main():
    await run_evaluation_pipeline()


if __name__ == "__main__":
    asyncio.run(main())