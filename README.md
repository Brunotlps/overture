# Overture

A **build-to-learn** project: a Q&A agent over a Git repository, built with FastAPI +
LangGraph + Docker.

**Status:** 🚧 Setup complete (Day 0). `/ask` returns `501` until the agent is built.

## Stack

- **Python 3.12 + `uv`** — fast, reproducible dependency management
- **FastAPI** — API layer
- **LangGraph** (core only, no full `langchain`) — agent orchestration
- **`langchain-openai`** — works with both OpenAI and local Ollama via configurable `base_url`
- **Docker** (multi-stage) + **Fly.io** — build and deploy
- **pytest + ruff** — tests and linting

## Running locally

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload
```

## Running with Docker

```bash
docker build -t overture:local .
docker run --rm -p 8000:8000 overture:local
```

## Endpoints

```bash
curl localhost:8000/health
# {"status": "ok", "version": "0.1.0"}

curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "where is circuit breaker implemented?"}'
# currently returns 501 - agent not implemented yet
```

## Tests

```bash
uv run pytest -v
```

Covers endpoint contracts (`test_health.py`) and agent tool guardrails — truncation of
large files, path traversal protection — (`test_tools.py`).

## Deploy

```bash
fly deploy
```

Live at `https://overture-prod.fly.dev`. Auto-deploy via GitHub Actions is intentionally
disabled until tests run as a required gate in CI.

## Notes

- No database, auth, or observability — out of scope.
- No LangGraph checkpointing yet — agent is stateless by design.
- `read_file` truncates files over 300 lines to protect LLM context.
