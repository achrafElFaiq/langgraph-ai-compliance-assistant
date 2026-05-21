# config/init_llm.py
import os
from langchain_openai import ChatOpenAI



# Agent LLM (générateur)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)

# Evaluator LLM (juge) — famille différente
evaluator_llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=8192,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)