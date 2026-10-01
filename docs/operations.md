# Operations

This document describes how the current service starts, deploys, logs, and fails.

## Startup

FastAPI uses the `lifespan` function in `app.main`.

Startup sequence:

1. Call `ensure_repo(settings.repo_path, settings.repo_git_url)` for the default repo.
2. Load curated repos from `settings.portfolio_repos_path` when the YAML file exists.
3. Build an in-memory `repo_registry` under `settings.repo_root`.
4. Populate `repo_display_names`.
5. Serve requests.

Default repo behavior differs from curated repo behavior:

| Repo kind | Failure behavior |
| --- | --- |
| Default `APP_REPO_PATH` with configured `APP_REPO_GIT_URL` | Clone failure raises `RuntimeError` and aborts startup. |
| Default `APP_REPO_PATH` without URL | Missing path logs `repo_missing`; tools will fail later if used. |
| Curated portfolio repo | Clone/materialization failure logs `portfolio_repo_skipped` and excludes that repo from `/repos`. |

Clone timeout is `120` seconds in `app.repo.CLONE_TIMEOUT_SECONDS`.

## Deployment

The production deployment target in `fly.toml` is:

- app: `overture-prod`;
- region: `gru`;
- internal port: `8000`;
- auto-stop enabled with zero minimum machines.

The Docker image:

- builds dependencies with `uv`;
- copies `app/` and the versioned `portfolio_repos.yaml` into the runtime image;
- installs `git` and `ca-certificates`;
- pre-clones the curated portfolio repos into `/data/repos/<repo_id>` during image build;
- starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

The curated repo list and curated repo contents are baked into the image so `/repos`
is available in production without mounting a separate YAML file or cloning every
portfolio repo on cold start. `ensure_repo()` treats those non-empty directories as
ready and logs them as existing repos.

Trade-off: baked portfolio repo contents are refreshed by building and deploying a
new image. The default target repo still follows `APP_REPO_PATH` and
`APP_REPO_GIT_URL`; production can point `APP_REPO_PATH` at one of the pre-cloned
paths, such as `/data/repos/codda`, to avoid a separate boot-time clone.

## CI/CD

GitHub Actions runs on pushes to `main` and pull requests:

```text
test:
  uv sync --locked
  uv run pytest
  uv run ruff check

deploy:
  flyctl deploy --remote-only
```

The deploy job runs only on push to `main` and only after `test` succeeds.

## Production Smoke Test

```bash
curl https://overture-prod.fly.dev/health

curl -X POST https://overture-prod.fly.dev/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $APP_API_KEY" \
  -d '{"question": "how is a new order created in this service?"}'
```

A healthy behavior answer should include at least one `read_file` step in the
response trajectory.

## Logs

All `app.*` logs are JSON lines emitted to stdout.

Key events:

| Event | Emitted by | Meaning |
| --- | --- | --- |
| `repo_ready` | `app.repo` | Existing repo path used. |
| `repo_missing` | `app.repo` | Repo path missing and no clone URL was configured. |
| `repo_cloned` | `app.repo` | Repo cloned successfully. |
| `repo_clone_failed` | `app.repo` | Repo clone failed. |
| `portfolio_repo_skipped` | `app.repo` | Curated repo excluded from registry. |
| `route_selected` | `app.graph` | ReAct route chosen after LLM decision. |
| `tool_executed` | `app.graph` | Tool call status and duration. |
| `budget_exceeded` | `app.graph` | Requested tool calls exceeded remaining budget. |
| `ask_completed` | `app.main` | Request completed with final state. |
| `ask_failed` | `app.main` | Graph invocation raised unexpectedly. |
| `summarization_failed` | `app.main` | History summarization failed; request continued without updating the summary. |
| `semantic_search_unavailable` | `app.semantic_search` | Embedding/index/search failed; semantic search returned no results. |

Each `/ask` request gets a `request_id` context variable attached to logs.

By default, logs contain allowlisted operational fields only. They preserve
`request_id`, error type, status, and duration where available. Set
`APP_LOG_DIAGNOSTICS_ENABLED=true` only for controlled debugging to include
clipped `question`, `tool_input`, error text, and stack traces. The diagnostic
limit is `APP_LOG_CONTENT_MAX_CHARS` (1–1000, default 200); stack traces are
capped at eight times that value. Git clone URLs and stderr are omitted from
application logs in both modes.

## Metrics and tracing

`GET /metrics` requires `X-API-Key` and exposes only Overture metrics from a
private Prometheus registry. The fixed `outcome` labels include `answered`,
`empty_answer_fallback`, `budget_exceeded`, `error`, `rejected`, and `other`.
Rejected requests count as `rejected` without a latency or iteration sample.
The metrics are:

- `overture_ask_requests_total` — accepted requests by outcome, plus quota
  rejections;
- `overture_ask_duration_seconds` — histogram for accepted request duration;
- `overture_ask_iterations` — histogram for tool calls per accepted request.

Example PromQL for five-minute p95 latency and the budget-exceeded rate:

```promql
histogram_quantile(0.95, sum by (le) (rate(overture_ask_duration_seconds_bucket[5m])))
sum(rate(overture_ask_requests_total{outcome="budget_exceeded"}[5m])) / sum(rate(overture_ask_requests_total{outcome!="rejected"}[5m]))
```

No scraper or long-term metrics store is configured in this repo. Counters are
per process and reset on restart; multiple instances must be scraped and
aggregated separately. Exact cost per request is not available because the app
does not yet collect complete provider token usage and pricing.

For tracing LLM and tool calls, LangSmith is available through the installed
LangChain stack as an explicit opt-in using `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY`. It can export prompts, repository content and answers, so
keep it disabled for ordinary production traffic unless that data is approved
for the chosen tracing service. We are not adding an OpenTelemetry exporter or
tracing collector without an operational need. See the
[LangChain tracing setup](https://docs.langchain.com/oss/python/integrations/llms/openai).

## Troubleshooting

| Symptom | Likely cause | Where to check |
| --- | --- | --- |
| `/ask` returns `503` | `APP_API_KEY` unset | `app.security.require_api_key` |
| `/ask` returns `401` | Missing or wrong `X-API-Key` | Client headers |
| `/ask` returns `429` | Per-client or global rate/concurrency quota exhausted | `Retry-After`, `APP_ASK_*` settings |
| `/ask` returns `413` | Chat input exceeds configured character budget | `APP_MODEL_MAX_INPUT_CHARS` |
| `/ask` returns `504` | Provider call timed out or request deadline exhausted | `APP_PROVIDER_TIMEOUT_SECONDS`, `APP_ASK_DEADLINE_SECONDS` |
| `/ask` returns `404` for `repo_id` | Repo was not registered at startup | `/repos`, startup logs |
| `/ask` returns `422` for `language` | Language is not one of `pt-BR` or `en` | `app.schemas.AskRequest`, `app.i18n` |
| Tools report repo path missing | `APP_REPO_PATH` missing and no clone URL/provisioned repo | `repo_missing` log |
| Startup aborts during clone | Configured default `APP_REPO_GIT_URL` failed or timed out | `repo_clone_failed` log |
| Cold start is slow or returns first-request 502s | Repos are being cloned during startup instead of reused from the image | Docker build logs, `repo_ready` startup logs |
| Answer says max tool calls reached | LLM requested more tools than `APP_MAX_ITERATIONS` allows | `budget_exceeded` log |
| Follow-up lost old context | Process restarted, history was summarized too aggressively, or summarization failed | `docs/api.md`, `summarization_failed` log |
| `/ask` returns `503` for storage | PostgreSQL unavailable, schema missing, or ownership metadata cannot be read | Check PostgreSQL availability and `APP_POSTGRES_*`; do not bypass ownership checks |
| `semantic_search` never appears in trajectory | `APP_SEMANTIC_SEARCH_ENABLED` is false or the model chose other tools | `app.config.Settings`, `app.agent_tools` |
| `semantic_search` returns no results | Embedding provider failed, repo has no eligible files, or index/search failed gracefully | `semantic_search_unavailable` log |

## PostgreSQL conversations

The default `APP_CHECKPOINTER_BACKEND=memory` keeps tests and local study offline.
For restarts and multiple Fly machines, provide a managed PostgreSQL database to
every instance, set `APP_CHECKPOINTER_BACKEND=postgres`, and store its connection
string in a server-side `APP_POSTGRES_DSN` secret. A local SQLite/file database
or Fly machine disk cannot share a thread across machines. Use a direct PostgreSQL
endpoint or a **session-mode** pooler: thread serialization holds a PostgreSQL
session advisory lock across each `/ask` call. Transaction-mode poolers cannot
preserve that lock. Configure the same authentication mode, principal IDs, and
resolved repository paths on every instance. Budget connections per process with
`APP_POSTGRES_POOL_MAX_SIZE` (default 20, minimum 4); each active request holds
one connection for its lock while the graph uses another for checkpoints.

Provision a dedicated database and back it up before enabling this backend.
Run one application instance with `APP_POSTGRES_SETUP=true` to execute LangGraph's
versioned checkpointer setup and create Overture's additive
`overture_conversations` table/index. After successful setup, set the flag to
`false` on all instances and deploy normally. Startup checks both schemas and
fails closed if they are unavailable. Do not run schema setup concurrently on
multiple instances or point the integration tests at production. Review upstream
LangGraph checkpoint migrations before a dependency upgrade; this app does not
run destructive data migration automatically.

The owner principal ID, resolved repository path, and last-use timestamp live in
`overture_conversations`; graph messages, summaries, and tool results live in
LangGraph's checkpoint tables. On a new thread, an existing checkpoint without
an ownership row is rejected rather than claimed. Existing in-memory threads
cannot be imported across a process restart. The serializer uses LangGraph's
MsgPack format with an explicit allowlist for Overture's stored types. Test a
backup with the new app version before changing those types or upgrading the
serializer. `durability="exit"` writes a completed turn on graph exit; unlike
memory mode, PostgreSQL retains prior checkpoints for a thread until it is
deleted. The message history inside each latest state is still summarized and
bounded by `APP_MAX_HISTORY_MESSAGES`.

Each request checks TTL before retrieving a checkpoint. Completed requests
opportunistically scan up to 100 expired and excess idle threads, skipping
threads locked by another instance. Thus `APP_THREAD_TTL_SECONDS` and
`APP_MAX_THREADS` are enforced on access and cleaned up progressively under
traffic; storage may remain above the cap when requests are in flight or no
requests arrive. Deletion removes checkpoint data before the ownership row so
a partial storage failure never makes old private state claimable by a new
principal. There is no public deletion endpoint. Operators needing targeted
erasure should stop writes for that thread and use an audited maintenance
procedure that deletes its checkpoints before its metadata row.

The application fails startup if PostgreSQL is unavailable. During a request,
database read/write errors return a generic `503` and do not fall back to
memory. An advisory-lock timeout returns `409`; retrying after the first request
finishes resumes the saved state. If a graph result was written before a later
storage error, the caller may receive `503`; retry with the same thread ID and
check the resulting conversation before submitting a non-idempotent request.
No DSN, principal ID, thread ID, or checkpoint content is included in application
error responses or metric labels.

External integration tests are opt-in:

```bash
OVERTURE_TEST_POSTGRES_DSN='postgresql://.../disposable_test_db' \
  uv run pytest tests/test_postgres_persistence.py
```

The default CI suite skips database integration tests but runs their offline
selection and failure tests. See the [LangGraph PostgreSQL checkpointer](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres)
for its migration and serializer contract.

## Operational Limitations

- PostgreSQL persistence is opt-in and needs an externally managed database.
- No persistent semantic-search index.
- No aggregated metrics store or default tracing backend; `/metrics` is process-local.
- Rate and concurrency limits are per process; no shared quota across instances.
- Individual principal keys require server-side provisioning; there is no rotation API.
- No persistent volume configured in `fly.toml`.
- Curated repo registry is built once at startup and is not mutated at runtime.
- Baked portfolio repo contents are frozen until the next image build/deploy.
- Answer language is request-scoped and limited to `pt-BR` and `en`.
