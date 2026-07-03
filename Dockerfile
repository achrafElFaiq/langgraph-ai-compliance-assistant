# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN pip install uv

COPY pyproject.toml uv.lock ./

# ── Serving image: prod deps only, runs the API (deployed) ──────────────
FROM base AS runtime
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
COPY . .
EXPOSE 8000
CMD [".venv/bin/uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Jobs image: prod + dev deps, runs ingestion and eval (offline) ──────
FROM base AS jobs
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
COPY . .
ENTRYPOINT ["python"]
