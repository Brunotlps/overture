# Operations

This document describes how the current service starts, deploys, logs, and fails.

## Startup

FastAPI uses the `lifespan` function in `app.main`.

Startup sequence:

1. Call `ensure_repo(settings.repo_path, settings.repo_git_url)` for the default repo.
2. Load optional curated repos from `settings.portfolio_repos_path`.
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
- copies only `app/` into the runtime image;
- installs `git` and `ca-certificates`;
- starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

Target repositories are not baked into the image.

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

`question` and `tool_input` are clipped to `APP_LOG_CONTENT_MAX_CHARS`.

## Troubleshooting

| Symptom | Likely cause | Where to check |
| --- | --- | --- |
| `/ask` returns `503` | `APP_API_KEY` unset | `app.security.require_api_key` |
| `/ask` returns `401` | Missing or wrong `X-API-Key` | Client headers |
| `/ask` returns `404` for `repo_id` | Repo was not registered at startup | `/repos`, startup logs |
| Tools report repo path missing | `APP_REPO_PATH` missing and no clone URL/provisioned repo | `repo_missing` log |
| Startup aborts during clone | Configured default `APP_REPO_GIT_URL` failed or timed out | `repo_clone_failed` log |
| Answer says max tool calls reached | LLM requested more tools than `APP_MAX_ITERATIONS` allows | `budget_exceeded` log |
| Follow-up lost old context | Process restarted, history was summarized too aggressively, or summarization failed | `docs/api.md`, `summarization_failed` log |
| `semantic_search` never appears in trajectory | `APP_SEMANTIC_SEARCH_ENABLED` is false or the model chose other tools | `app.config.Settings`, `app.agent_tools` |
| `semantic_search` returns no results | Embedding provider failed, repo has no eligible files, or index/search failed gracefully | `semantic_search_unavailable` log |

## Operational Limitations

- No persistent checkpointer.
- No persistent semantic-search index.
- No metrics or tracing backend.
- No rate limiting.
- No per-client credentials.
- No persistent volume configured in `fly.toml`.
- Curated repo registry is built once at startup and is not mutated at runtime.
