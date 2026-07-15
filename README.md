# Overture

A **build-to-learn** project: a Q&A agent over a Git repository, built with FastAPI +
LangGraph + Docker.

**Status:** `/ask` is backed by a ReAct-style LangGraph agent with tool calling over a
configured repository.

## Stack

- **Python 3.12 + `uv`** — fast, reproducible dependency management
- **FastAPI** — API layer
- **LangGraph** (core only, no full `langchain`) — agent orchestration
- **`langchain-openai`** — works with both OpenAI and local Ollama via configurable `base_url`
- **Docker** (multi-stage) + **Fly.io** — build and deploy
- **pytest + ruff** — tests and linting

## Agent design

The public API accepts a natural-language `question`. The ReAct graph lets the LLM
decide whether to answer directly or call repository tools such as `read_file`,
`list_files`, and `grep_repo`.

The previous deterministic category graph is still present as `build_graph()` for
study and regression tests. The application composition root now uses
`build_react_graph()`.

## Running locally

```bash
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload
```

Relevant environment variables:

```bash
APP_LLM_MODEL=gpt-4o-mini
APP_LLM_BASE_URL=https://api.openai.com/v1
APP_LLM_API_KEY=...
APP_REPO_PATH=/path/to/repository
```

`APP_REPO_PATH` points to the Git repository the tools should inspect. The settings use
the `APP_` prefix, so `repo_path` is configured as `APP_REPO_PATH`.

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
```

`/ask` only accepts `question`. Legacy request fields such as `target` are rejected with
`422`; tool arguments are now chosen by the ReAct agent.

## Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check
```

`UV_CACHE_DIR=.uv-cache` keeps uv's cache inside the workspace, which avoids sandbox
or permission issues with a global cache.

The tests cover endpoint contracts, deterministic tool guardrails, the legacy
deterministic graph, and the ReAct graph/tool-calling loop.

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
