from dotenv import load_dotenv
load_dotenv()

import uvicorn
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.init_store import store
from src.api.routes import chat, health, stream, admin
from contextlib import asynccontextmanager
from src.config.settings import setup_logging




@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    await store.connect()
    yield
    await store.close()

app = FastAPI(
    title="Compliance Assistant API",
    description="RAG-based compliance assistant for EU regulations",
    version="1.0.0",
    lifespan=lifespan,
)


origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(stream.router)
app.include_router(admin.router)

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)