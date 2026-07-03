# config/init_llm.py
import os
from langchain_openai import ChatOpenAI


# Agent LLM (générateur)
llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=8192,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)


critic_llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=8192,
    temperature=0,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)


grounder_llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    max_tokens=512,
    temperature=0,
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)

