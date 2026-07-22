# Overture

A **build-to-learn** project: a Q&A agent over a Git repository, built with FastAPI +
LangGraph + Docker.

**Status:** `/ask` runs on a ReAct-style LangGraph agent. The LLM decides, guided by a
system prompt with an explicit investigation strategy, whether to answer directly or
call repository tools (`list_files`, `read_file`, `grep_repo`) under a tool-call budget.
The service runs in production on Fly.io behind an API key, deployed automatically by a
CI pipeline that requires tests and lint to pass first.

## Stack

- **Python 3.12 + `uv`** — fast, reproducible dependency management
- **FastAPI** — API layer
- **LangGraph** (core only, no full `langchain`) — agent orchestration
- **`langchain-openai`** — works with both OpenAI and local Ollama via configurable `base_url`
- **Docker** (multi-stage) + **Fly.io** — build and deploy
- **GitHub Actions** — CI gate (pytest + ruff) and auto-deploy gated on green tests
- **pytest + ruff** — tests and linting

## Documentation

The root README is the quick entry point. Detailed technical documentation lives in
[`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md) - local setup, environment variables,
  Docker, and common commands.
- [API](docs/api.md) - endpoint contracts for `/health`, `/repos`, and `/ask`.
- [Architecture](docs/architecture.md) - components, ReAct graph flow, repository
  tools, and trade-offs.
- [Operations](docs/operations.md) - startup provisioning, deploy, logs, and
  troubleshooting.
- [Testing and eval](docs/testing-and-eval.md) - pytest, ruff, CI, and the LLM eval
  harness.
- [Security](docs/security.md) - API key auth, tool guardrails, logging controls, and
  remaining risks.

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
APP_API_KEY=dev-key  # clients must send this in the X-API-Key header
```

`APP_REPO_PATH` points to the Git repository the tools should inspect (settings use the
`APP_` prefix, so `repo_path` becomes `APP_REPO_PATH`). Then:

```bash
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key" \
  -d '{"question": "how is a new order created in this service?"}'
```

`/ask` accepts `question` plus the optional `thread_id` and `repo_id` fields described
below. Legacy request fields such as `target` are rejected with `422`; tool arguments
are chosen by the ReAct agent.

## Authentication

`/ask` requires an API key in the `X-API-Key` header, validated against `APP_API_KEY`
(fail-closed: if the server has no key configured, every request is rejected with
`503`; a missing or wrong key returns `401` before the agent runs, so no LLM tokens
are spent). `/health` stays public for platform health checks.

## Conversation memory

`/ask` accepts an optional `thread_id`. Omit it for a stateless, one-off question
(default behavior). Reuse the `thread_id` returned by a previous response to continue
that conversation — the agent sees prior questions and answers, so follow-ups like
"and where is that function called?" work.

Two things to know:

- **In-memory only** — conversations are held in the process's memory
  (`MemorySaver`), not a database. They do not survive a restart or, on Fly's
  scale-to-zero, a machine going idle. This is a deliberate scope choice for a
  study project; production use would need a persistent checkpointer.
- **Bounded via summarization** — once a conversation passes `APP_MAX_HISTORY_MESSAGES`
  (default 20) messages, the oldest ones are folded into a rolling `conversation_summary`
  (an LLM call over the messages being dropped, combined with any prior summary so it
  stays a single updated summary rather than a growing list) instead of being discarded
  outright; the summary is injected into the system prompt on later turns. It is
  defensively capped at a fixed length so it can't grow unbounded call after call. If
  the summarization call itself fails, that turn falls back to dropping the messages
  without updating the summary, so `/ask` never fails because of it. The per-question
  tool-call budget (`APP_MAX_ITERATIONS`) always resets at the start of each turn,
  regardless of history length.

## Portfolio repos

Beyond the single `APP_REPO_PATH` default repo, the app can serve a small, curated set
of additional repos — meant for a portfolio frontend where a visitor picks one of
several showcased projects before asking questions.

- **`portfolio_repos.yaml`** (path configurable via `APP_PORTFOLIO_REPOS_PATH`) lists
  the curated repos: `repo_id`, `git_url`, `display_name`. The file is optional — if
  it's absent, this feature is simply off and behavior is unchanged.
- On startup, each entry is shallow-cloned into `{APP_REPO_ROOT}/{repo_id}/`. A repo
  that fails to clone is logged and excluded from the registry rather than aborting
  startup — one broken portfolio entry shouldn't take down the whole app (unlike the
  single required `APP_REPO_PATH` repo, which still aborts startup on failure).
- **`GET /repos`** (same `X-API-Key` auth as `/ask`) lists the repos that registered
  successfully, as `{repo_id, display_name}` pairs — enough for a frontend to render a
  project picker.
- **`AskRequest.repo_id`** — omit it to use the default `APP_REPO_PATH` repo (today's
  behavior); set it to a `repo_id` from `GET /repos` to target that repo instead. An
  unknown `repo_id` returns `404`.
- The registry is built once at startup and never mutated at runtime — no dynamic
  registration by request-time URL, no quota/eviction logic, since the set of repos is
  small and decided ahead of time by whoever configures the YAML, not by callers.
- `thread_id` and `repo_id` aren't cross-validated: nothing stops a conversation from
  switching `repo_id` mid-thread. Not enforced for now — simple enough to skip until it
  becomes a real problem.

## Semantic search

`grep_repo` is exact substring matching — it misses conceptual questions with no
literal term to search for (e.g. "how is money handled?" over code that only says
`total_price`). `APP_SEMANTIC_SEARCH_ENABLED` (default `false`) adds a `semantic_search`
tool alongside it, built on OpenAI embeddings:

- **Whole-file embeddings** — one embedding per eligible file (same sensitive-path and
  binary-file filtering as the other tools), not per-function or per-chunk. Simple, and
  good enough since the tool only needs to *locate* a candidate file — the agent still
  calls `read_file` to confirm before answering.
- **Lazy, per-repo, in-memory index** — the index for a repo is built on its first
  `semantic_search` call, not at startup, and cached for the process's lifetime. With
  multiple portfolio repos, eager indexing at startup would mean paying embedding costs
  on every cold start for repos nobody queries semantically.
- **Degrades gracefully** — if the embedding provider fails (rate limit, network error),
  `semantic_search` returns an empty result instead of failing the request; the agent
  falls back to `grep_repo`.

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

The log level is configurable via `APP_LOG_LEVEL` (default `INFO`). User-provided
content (`question`, `tool_input`) is truncated to `APP_LOG_CONTENT_MAX_CHARS`
(default 200) before logging, so no log field grows unboundedly with user input.

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

The tests cover endpoint contracts, API-key authentication (rejected requests never
reach the graph), tool guardrails (path traversal, sensitive files, binaries,
truncation), startup repository provisioning, structured logging, conversation
memory (thread continuation, per-turn budget reset, summary-backed history
compaction), optional semantic search, the legacy deterministic graph, and the ReAct
graph/tool-calling loop.

## Answer quality eval

`eval/` holds a small local harness that measures agent answer quality against
a fixed, version-controlled fixture repo (`eval/fixture_repo/`) — it is
separate from pytest because it calls a real LLM, so it costs money and can be
flaky; it is not run in CI.

```bash
uv run python -m eval.run
```

It runs the question set in `eval/cases.py` against the ReAct graph and prints
a report: answered rate, budget-exceeded rate, expected-tool presence, and
average iterations, so two runs (e.g. before/after a prompt change) can be
compared directly.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and pull request:

- **test** — `uv run pytest` + `uv run ruff check` (no LLM secrets needed; tests use
  fakes).
- **deploy** — `flyctl deploy` to production, only on push to `main` and only after
  `test` passes.

`main` is protected: changes land via pull request, and the `test` check is required
before merging. A red build never deploys.

Live at `https://overture-prod.fly.dev`. Manual deploys are still possible with
`fly deploy` for emergencies.

### Production smoke test

After a deploy, verify the service end to end:

```bash
curl https://overture-prod.fly.dev/health
# {"status": "ok", "version": "0.1.0"}

curl -X POST https://overture-prod.fly.dev/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"question": "how is a new order created in this service?"}'
# expect an answer grounded in files, with a trajectory of tool calls
```

The `trajectory` field in the response shows which tools ran — a healthy answer to a
behavior question should include a `read_file` step, not just `grep_repo`.

## Known limitations

- **Conversation memory is in-memory only** — see [Conversation memory](#conversation-memory);
  it does not survive a restart or scale-to-zero. Old turns are summarized before
  being dropped when possible, but the summary is still process-local.
- **Repos are curated, not arbitrary** — `/ask` can target the default repo or one of
  the repos listed in `portfolio_repos.yaml` (see [Portfolio repos](#portfolio-repos)),
  all fixed at startup; there is no API to register or clone an arbitrary repo at
  request time.
- **Answer quality depends on the model's tool calling** — the system prompt guides
  investigation, but a weak tool-calling model can still answer from grep snippets or
  waste the tool budget.
- **Single static API key** — one shared key with no per-client rotation or rate
  limiting; a leaked key must be rotated manually (`fly secrets set APP_API_KEY=...`).
- **Minimal observability** — logs only; no metrics, tracing, or structured trajectory
  export beyond the API response.
