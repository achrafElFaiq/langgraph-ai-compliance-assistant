from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI
from src.config.init_store import store
from src.api.routes import chat, health

app = FastAPI(
    title="Compliance Assistant API",
    description="RAG-based compliance assistant for EU regulations",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(chat.router)

@app.on_event("startup")
async def startup():
    await store.connect()

@app.on_event("shutdown")
async def shutdown():
    await store.close()

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)