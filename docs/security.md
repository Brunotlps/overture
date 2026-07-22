# Security

Overture's current security posture is intentionally small in scope: protect the
paid LLM path with a static API key, prevent repository tools from exposing obvious
sensitive files or escaping the target repo, and avoid leaking raw exceptions to
clients.

## API Authentication

`/ask` and `/repos` use `require_api_key` from `app.security`.

Behavior:

- no server-side `APP_API_KEY`: return `503`;
- missing or wrong `X-API-Key`: return `401`;
- valid key: continue to route handler.

The comparison uses `secrets.compare_digest`.

`/health` is public for platform health checks.

## Request Validation

`AskRequest` in `app.schemas`:

- requires `question` length from 3 to 500 characters;
- allows optional `thread_id` up to 100 characters;
- allows optional `repo_id`;
- forbids unknown fields.

An unknown `repo_id` returns `404` before graph invocation.

## Repository Tool Guardrails

`app.tools` applies the following controls:

- path resolution must remain within the target repository;
- ignored directories are skipped/rejected: `.git`, `.claude`, `__pycache__`, `node_modules`, `.venv`;
- sensitive file patterns are blocked: `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `*credentials*`, `*secret*`, `*token*`;
- binary files are skipped or rejected;
- `read_file` rejects directories with a clear "not a file" error;
- large files are truncated after 300 lines;
- grep output is limited to 20 matches by default and 200 characters per matching line.

Tests cover traversal, absolute path escape, ignored directories, sensitive files,
binary files, directory targets, and truncation in `tests/test_tools.py`.

When `APP_SEMANTIC_SEARCH_ENABLED=true`, semantic search reuses `list_files` and
`read_file` for eligible files, so the same path, sensitive-file, binary-file, and
truncation guardrails apply before content is embedded or returned as snippets.

## Logging Controls

`app.observability.clip` truncates logged `question` and `tool_input` values to
`APP_LOG_CONTENT_MAX_CHARS`.

`/ask` client-facing 500 responses use a generic detail:

```text
Unexpected error running the agent
```

The full exception is logged in `ask_failed` for debugging.

## Multi-repo Design

The implemented multi-repo feature is config-driven:

- visitors can choose a configured `repo_id`;
- visitors cannot submit a `git_url`;
- no runtime `POST /repos` endpoint exists.

This design was chosen after issue #21, which described a dynamic registration
endpoint with SSRF concerns, was superseded by issue #23's curated portfolio scope.

## Remaining Risks

| Risk | Status |
| --- | --- |
| Static shared API key | Implemented but coarse-grained; no per-client identity or rotation API. |
| No rate limiting | A leaked valid key can spend LLM tokens until manually rotated. |
| Curated YAML trust boundary | `git_url` values are trusted configuration, not user input. |
| Logs still include clipped user content | Truncation bounds size but does not fully redact content. |
| Conversation memory and summaries in process | No durable store, no encryption-at-rest concerns inside this app, but no persistence guarantees. |
| Semantic search sends file content to embedding provider | Only eligible non-sensitive files are embedded, but repo content still leaves the process when the feature is enabled. |
| Repository content exposure | Tools expose non-sensitive text files from configured repos to the LLM and response trajectory summaries. |

## Not Implemented

- OAuth or per-user authentication.
- Authorization by repo or client.
- Rate limiting or quotas.
- SSRF allowlist for caller-submitted URLs, because caller-submitted URLs are not supported.
- Secret scanning beyond filename pattern filtering.
- Metrics/tracing with privacy controls.
