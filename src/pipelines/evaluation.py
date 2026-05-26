"""
"# run_eval.py
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
from ragas.llms import LangchainLLMWrapper
from src.infrastructure.eval.ragas_judge import RagasJudge
from src.config.init_llm import evaluator_llm
from src.application.agent.graph import compiled_graph
from src.config.init_store import store


async def main():
    with open("datasets/eval/dataset.json", "r") as f:
        dataset = json.load(f)
    await store.connect()

    judge = RagasJudge(evaluator_llm=evaluator_llm)
    result = await judge.eval(dataset=dataset, agent=compiled_graph)

    await store.close()

    print(f"Faithfulness:  {result.faithfulness}")
    print(f"Factual:       {result.factual_correctness}")
    print(f"Ctx Recall:    {result.context_recall}")
    print(f"Ctx Precision: {result.context_precision}")


if __name__ == "__main__":
    asyncio.run(main())
 """

# run_eval.py
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
import os
from src.infrastructure.eval.deepeval_judge import DeepEvalJudge
from src.infrastructure.eval.deepeval_judge import OpenRouterDeepEvalLLM
from src.application.agent.graph import compiled_graph
from src.config.init_store import store


async def main():
    with open("datasets/eval/dataset.json", "r") as f:
        dataset = json.load(f)
    await store.connect()

    model = OpenRouterDeepEvalLLM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_tokens=8192
    )
    judge = DeepEvalJudge(model=model)
    result = await judge.eval(dataset=dataset, agent=compiled_graph)

    await store.close()

    print(f"Faithfulness:  {result.faithfulness}")
    print(f"Factual:       {result.factual_correctness}")
    print(f"Ctx Recall:    {result.context_recall}")
    print(f"Ctx Precision: {result.context_precision}")


if __name__ == "__main__":
    asyncio.run(main())