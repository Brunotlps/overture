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
- allows optional `language`, restricted to `pt-BR` or `en` and defaulting to `pt-BR`;
- forbids unknown fields.

An unknown `repo_id` returns `404` before graph invocation.
Unsupported `language` values return `422` during request validation.

## Repository Tool Guardrails

`app.tools` applies the following controls:

- path resolution must remain within the target repository;
- symlinks in requested paths are rejected, including links to files inside the repository;
- ignored directories are skipped/rejected regardless of case: `.git`, `.claude`, `__pycache__`, `node_modules`, `.venv`;
- sensitive file patterns are blocked in any path component, regardless of case: `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `*credentials*`, `*secret*`, `*token*`;
- binary files are skipped or rejected;
- `read_file` rejects directories with a clear "not a file" error;
- `read_file` streams a numbered line range (`start_line`, `max_lines` up to 300) instead of loading the whole file, caps each line at 2,000 characters and each response at 20,000 characters, and ends with the `start_line` to continue when more lines follow;
- `grep_repo` streams files with the same per-line cap, so text past 2,000 characters on a single line is not searched;
- grep output is limited to 20 matches by default and 200 characters per matching line.

Tests cover traversal, absolute path escape, symlinks, ignored directories,
sensitive files, binary files, directory targets, line ranges, output caps, and
bounded memory in `tests/test_tools.py`.

When `APP_SEMANTIC_SEARCH_ENABLED=true`, semantic search reuses `list_files` and
the same eligibility checks as `read_file`, so the same path, sensitive-file, and
binary-file guardrails apply before content is embedded or returned as snippets.
Indexing admits at most 1,000 files, 1 MB per file, and 20 MB in total, streams at
most 300 lines and 20,000 characters per file, and embeds in batches of 100.
Filename filtering is not secret scanning. The checks operate on filesystem paths
at read time; repositories that can change concurrently require a stronger
filesystem boundary to prevent a file from being replaced between validation
and opening.

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
