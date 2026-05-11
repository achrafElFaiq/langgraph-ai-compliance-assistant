import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek/deepseek-v4-pro",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={"HTTP-Referer": "http://localhost", "X-Title": "compliance-assistant"}
)