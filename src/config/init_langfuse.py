from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

langfuse_client = Langfuse()

# Stateless in v4: reads the active trace context, so a single shared instance
# nests every LangChain/LangGraph step under whatever observation is current.
langfuse_handler = CallbackHandler()
