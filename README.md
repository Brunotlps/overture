# Overture

A **build-to-learn** project: a Q&A agent over a Git repository, built with FastAPI +
LangGraph + Docker.

**Status:** `/ask` runs on a ReAct-style LangGraph agent. The LLM decides, guided by a
system prompt with an explicit investigation strategy, whether to answer directly or
call repository tools (`list_files`, `read_file`, `grep_repo`) under a tool-call budget.

## Stack

- **Python 3.12 + `uv`** — fast, reproducible dependency management
- **FastAPI** — API layer
- **LangGraph** (core only, no full `langchain`) — agent orchestration
- **`langchain-openai`** — works with both OpenAI and local Ollama via configurable `base_url`
- **Docker** (multi-stage) + **Fly.io** — build and deploy
- **pytest + ruff** — tests and linting

## Agent design

The public API accepts a natural-language `question`. The ReAct graph
(`build_react_graph()`) is the current runtime: the LLM loops between deciding and
executing tools until it produces a final answer or exhausts the tool budget
(`APP_MAX_ITERATIONS`, default 5). The system prompt instructs the model to read
implementation files before answering behavior questions (grep only locates), to retry
naming variants when a search misses, and reports the remaining tool budget on every
iteration so the model answers before hitting the guardrail.

Tool hardening: binary files are skipped/rejected, grep matches are truncated to 200
chars per line, files are truncated at 300 lines, and sensitive paths plus ignored
directories (`.git`, `.claude`, etc.) are never exposed.

The previous deterministic category graph is still present as `build_graph()` for
study and regression tests.

## Target repository

The agent answers questions about the repository at `APP_REPO_PATH`. On startup, if
that path is missing or empty and `APP_REPO_GIT_URL` is set, the app shallow-clones
the URL into `APP_REPO_PATH` (production uses
[`Brunotlps/codda`](https://github.com/Brunotlps/codda)). If the path already has
content, it is used as-is — so local development can keep pointing `APP_REPO_PATH` at
any local clone (`target_repo/` is git-ignored and provisioned manually). A configured
URL that fails to clone aborts startup instead of serving an agent with broken tools.

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
APP_REPO_GIT_URL=https://github.com/Brunotlps/codda  # optional: clone target at startup
```

`APP_REPO_PATH` points to the Git repository the tools should inspect (settings use the
`APP_` prefix, so `repo_path` becomes `APP_REPO_PATH`). Then:

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "how is a new order created in this service?"}'
```

`/ask` only accepts `question`. Legacy request fields such as `target` are rejected with
`422`; tool arguments are chosen by the ReAct agent.

## Observability

Every `app.*` log is emitted to stdout as one JSON line (ready for a log collector),
correlated by a per-request `request_id`. Events:

- `route_selected` — the route chosen after each LLM decision (`execute_tools`,
  `finalize`, `budget_exceeded`) with the requested tools and iteration count.
- `tool_executed` — one per tool call, with `tool`, `tool_input`, `status`
  (`ok`/`error`/`unknown_tool`) and `duration_ms`. Failures log at `WARNING`.
- `ask_completed` — one per request, with `question`, `tools_called`, `iterations`,
  `outcome` (`answered`, `empty_answer_fallback`, `budget_exceeded`) and `duration_ms`.
- `ask_failed` — logged at `ERROR` with the exception and stack trace when the graph
  raises.

The log level is configurable via `APP_LOG_LEVEL` (default `INFO`).

Example line:

```json
{"timestamp": "2026-07-16T20:01:26.588+00:00", "level": "INFO", "logger": "app.main", "event": "ask_completed", "request_id": "f6a63711251e42f6baf1f760fcaf658b", "question": "What files exist?", "tools_called": ["list_files", "agent_decide"], "iterations": 1, "outcome": "answered", "duration_ms": 9.5}
```

## Running with Docker

```bash
docker build -t overture:local .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e APP_REPO_PATH=/data/repo \
  -e APP_REPO_GIT_URL=https://github.com/Brunotlps/codda \
  overture:local
```

The image does not bundle a target repository — it is shallow-cloned from
`APP_REPO_GIT_URL` at startup. To use a local repository instead, mount it
(`-v /path/to/repository:/data/repo`) and omit `APP_REPO_GIT_URL`.

## Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check
```

`UV_CACHE_DIR=.uv-cache` keeps uv's cache inside the workspace, which avoids sandbox
or permission issues with a global cache.

The tests cover endpoint contracts, tool guardrails (path traversal, sensitive files,
binaries, truncation), the legacy deterministic graph, and the ReAct
graph/tool-calling loop.

## Deploy

Deploy is **manual**:

```bash
fly deploy
```

Live at `https://overture-prod.fly.dev`. CI/CD (GitHub Actions auto-deploy) is
intentionally disabled: there is no CI gate running the test suite yet, so pushing to
`main` must not ship straight to production. It will be re-enabled once tests are a
required check.

### Production smoke test

After a deploy, verify the service end to end:

```bash
curl https://overture-prod.fly.dev/health
# {"status": "ok", "version": "0.1.0"}

curl -X POST https://overture-prod.fly.dev/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "how is a new order created in this service?"}'
# expect an answer grounded in files, with a trajectory of tool calls
```

The `trajectory` field in the response shows which tools ran — a healthy answer to a
behavior question should include a `read_file` step, not just `grep_repo`.

## Known limitations

- **No checkpointing** — the agent is stateless; every request starts a fresh graph run
  with no memory of previous questions.
- **Target repo is fixed per deployment** — the repository is chosen at startup
  (`APP_REPO_GIT_URL` or a manually provisioned path); there is no API to point the
  agent at an arbitrary repo at request time.
- **Answer quality depends on the model's tool calling** — the system prompt guides
  investigation, but a weak tool-calling model can still answer from grep snippets or
  waste the tool budget.
- **No authentication** — `/ask` is publicly reachable in production; anyone with the
  URL can spend LLM tokens.
- **Minimal observability** — logs only; no metrics, tracing, or structured trajectory
  export beyond the API response.
