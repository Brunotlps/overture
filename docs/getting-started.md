# Getting Started

This guide documents the local workflow that matches the current codebase.

## Requirements

- Python 3.12, matching `.python-version`.
- `uv`, used for dependency and command execution.
- `git`, required when `APP_REPO_GIT_URL` is used to clone a target repository at startup.
- An OpenAI-compatible chat model endpoint for real `/ask` calls.

Dependencies are declared in `pyproject.toml`; the lockfile is `uv.lock`.

## Install

```bash
uv sync
cp .env.example .env
```

Set at least:

```bash
APP_LLM_API_KEY=...
APP_API_KEY=dev-key
APP_REPO_PATH=/path/to/repository
```

In default shared mode, `APP_API_KEY` is required for `/ask`, `/repos`, and
`/metrics`; if unset, those endpoints return `503`. Individual mode instead
requires `APP_PRINCIPAL_API_KEYS` (see [Security](security.md)).

## Important Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL used by `ChatOpenAI`. |
| `APP_LLM_MODEL` | `gpt-4o-mini` | Chat model used by the graph. |
| `APP_LLM_API_KEY` | `changeme` | API key passed to the model client. |
| `APP_MAX_ITERATIONS` | `5` | Per-question tool-call budget. |
| `APP_REPO_PATH` | `/data/repo` | Default repository inspected by tools. |
| `APP_REPO_GIT_URL` | empty | Optional URL cloned into `APP_REPO_PATH` at startup. |
| `APP_API_KEY` | empty | Shared study mode's static API key in `X-API-Key`. |
| `APP_AUTH_MODE` | `shared` | `shared` study principal or `individual` principals. |
| `APP_PRINCIPAL_API_KEYS` | `{}` | JSON mapping of principal IDs to distinct secret keys, required in individual mode. |
| `APP_ASK_RATE_PER_CLIENT` / `APP_ASK_RATE_GLOBAL` | `30` / `120` | Accepted `/ask` requests per sliding window, per peer IP and globally in this process. |
| `APP_ASK_RATE_WINDOW_SECONDS` | `60` | Sliding-window length. |
| `APP_ASK_CONCURRENCY_PER_CLIENT` / `APP_ASK_CONCURRENCY_GLOBAL` | `2` / `8` | Active `/ask` limits per peer IP and globally in this process. |
| `APP_ASK_DEADLINE_SECONDS` | `60` | Remaining time checked before each provider call. |
| `APP_PROVIDER_TIMEOUT_SECONDS` / `APP_PROVIDER_MAX_RETRIES` | `20` / `0` | Per-call timeout and retry configuration for chat and embeddings. |
| `APP_MODEL_MAX_COMPLETION_TOKENS` / `APP_MODEL_MAX_INPUT_CHARS` | `1024` / `100000` | Requested chat output cap and input character budget for chat, summary, and embedding batches. |
| `APP_MAX_HISTORY_MESSAGES` | `20` | Number of historical messages kept per thread before older turns are summarized and dropped. |
| `APP_THREAD_TTL_SECONDS` | `86400` | Idle time after which a conversation expires. Must be positive. |
| `APP_MAX_THREADS` | `500` | Maximum retained conversations; least recently used idle threads are evicted. Must be positive. |
| `APP_CHECKPOINTER_BACKEND` | `memory` | `memory` for local use or `postgres` for shared durable storage. |
| `APP_POSTGRES_DSN` | empty | Server-held PostgreSQL connection string, required for `postgres`. |
| `APP_POSTGRES_SETUP` | `false` | Explicitly run additive table setup and LangGraph migrations at startup. Use once, then disable. |
| `APP_POSTGRES_POOL_MAX_SIZE` | `20` | Maximum database connections per app process for the checkpointer and thread locks. |
| `APP_PORTFOLIO_REPOS_PATH` | `portfolio_repos.yaml` | Optional curated repo YAML path. |
| `APP_REPO_ROOT` | `/data/repos` | Parent directory for curated repo clones. |
| `APP_LOG_LEVEL` | `INFO` | Log level for `app.*` loggers. |
| `APP_LOG_CONTENT_MAX_CHARS` | `200` | Maximum logged length of questions and tool inputs. |
| `APP_SEMANTIC_SEARCH_ENABLED` | `false` | Adds the `semantic_search` tool backed by OpenAI embeddings when enabled. |

The settings class is `app.config.Settings`.

## Run Locally

```bash
uv run uvicorn app.main:app --reload
```

Then call:

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"question": "What files are in this repository?"}'
```

## Target Repository Setup

There are two supported local patterns:

1. Point `APP_REPO_PATH` at an existing local clone.
2. Set `APP_REPO_GIT_URL`; startup will shallow-clone it into `APP_REPO_PATH` if the path is missing or empty.

If `APP_REPO_PATH` exists and has content, it is used as-is. If `APP_REPO_GIT_URL`
is configured and cloning fails, startup aborts. If no URL is configured and the
path is missing, startup logs a warning and the agent tools will fail when used.

## Portfolio Repos

`APP_PORTFOLIO_REPOS_PATH` defaults to the versioned `portfolio_repos.yaml`. To use
a different curated set, create a YAML file with the same shape and point the
setting at it:

```yaml
repos:
  - repo_id: overture
    git_url: https://github.com/Brunotlps/overture
    display_name: Overture
```

`repo_id` must match `^[a-z0-9][a-z0-9-]*$`. The registry is built once at
startup. Failed curated repo clones are logged and skipped instead of taking down
the service.

The default file includes four curated portfolio projects:

| `repo_id` | Display name | Git URL |
| --- | --- | --- |
| `overture` | Overture | `https://github.com/Brunotlps/overture` |
| `codda` | Codda | `https://github.com/Brunotlps/codda` |
| `briskmail` | BriskMail | `https://github.com/Brunotlps/email-classifier` |
| `interlude` | Interlude | `https://github.com/Brunotlps/interlude` |

## Docker

```bash
docker build -t overture:local .
docker run --rm -p 8000:8000 \
  --env-file .env \
  -e APP_REPO_PATH=/data/repo \
  -e APP_REPO_GIT_URL=https://github.com/Brunotlps/codda \
  overture:local
```

The image includes the versioned `portfolio_repos.yaml` and pre-clones its curated
repos into `/data/repos/<repo_id>` during build, so `/repos` can expose the curated
list in production without cloning every portfolio repo on cold start.

The default target repo still follows `APP_REPO_PATH`. To reuse a pre-cloned
portfolio repo as the default target in a container, set `APP_REPO_PATH` to one of
those paths, for example `/data/repos/codda`. To use a different local repository,
mount it and point `APP_REPO_PATH` at the mount.

## Answer Language

`/ask` defaults to Brazilian Portuguese answers:

```json
{
  "question": "O que este projeto faz?"
}
```

Pass `language: "en"` for English:

```json
{
  "question": "What does this project do?",
  "language": "en"
}
```

Only `pt-BR` and `en` are accepted. Unsupported values return `422`.

## Optional Semantic Search

Enable semantic search with:

```bash
APP_SEMANTIC_SEARCH_ENABLED=true
```

When enabled, the agent receives a fourth tool, `semantic_search`, for conceptual
questions where exact `grep_repo` terms miss. The tool builds a lazy, per-repo,
in-memory embedding index on the first semantic search for that repo. It uses the
same OpenAI-compatible base URL and API key settings as the chat model.

Embedding/index failures do not fail `/ask`; the tool returns no results and the
agent can fall back to `grep_repo` or other tools.

## Quality Commands

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
UV_CACHE_DIR=.uv-cache uv run ruff check
```

`UV_CACHE_DIR=.uv-cache` keeps uv cache writes inside the repository workspace.
