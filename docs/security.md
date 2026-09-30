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

By default, `app.observability.JsonFormatter` emits allowlisted operational
fields per event. Questions, tool arguments, exception messages, repository
paths, Git URLs, and stack traces are omitted. A `request_id` is captured when
the log record is created, so deferred formatting keeps request correlation.
Error events retain error type, status, and duration where available. Clone
failures omit both the URL and Git stderr; credentials are never logged through
these application events.

`APP_LOG_DIAGNOSTICS_ENABLED=true` explicitly enables clipped question,
tool-argument, error, and stack-trace fields for controlled diagnostics.
`APP_LOG_CONTENT_MAX_CHARS` defaults to 200 and accepts 1–1000 characters;
stack traces are capped at eight times that limit. Diagnostic logs may contain
secrets and should be handled accordingly. The app cannot control logs emitted
directly by third-party libraries outside the `app.*` logger.

Expected repository tool failures use stable codes (`file_not_found`,
`invalid_input`, `filesystem_error`) and generic recovery guidance in
`ToolMessage` and the public trajectory. Raw exception text does not enter
those messages. The normal trajectory still includes the tool arguments by
API contract; these may contain content submitted by the caller or model.

`/ask` client-facing 500 responses use a generic detail:

```text
Unexpected error running the agent
```

The full exception is available only when diagnostic logging is explicitly enabled.

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
| Diagnostic logs may include sensitive content | Enable only for controlled troubleshooting; private mode is the default. |
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
