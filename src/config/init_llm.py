# config/init_llm.py
import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas.llms import llm_factory


# Agent LLM (générateur)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=8192,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)


critic_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-lite-001",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=8192,
    temperature=0,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)


grounder_llm = ChatOpenAI(
    model="google/gemini-2.0-flash-lite-001",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=4096,
    temperature=0,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)

# Evaluator LLM (juge) — famille différente
evaluator_llm = llm_factory(
    model="openai/gpt-4o-mini",
    client=OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
    ),
    max_tokens=8192
)


