# Security

Overture's security posture is intentionally small in scope: protect the
paid LLM path with API keys, prevent repository tools from exposing obvious
sensitive files or escaping the target repo, and avoid leaking raw exceptions to
clients.

## API Authentication

`/ask`, `/repos`, and `/metrics` use `require_api_key` from `app.security`.

Behavior in the default `APP_AUTH_MODE=shared` study deployment:

- no server-side `APP_API_KEY`: return `503`;
- missing or wrong `X-API-Key`: return `401`;
- valid key: continue to route handler.

The comparison uses `secrets.compare_digest`.

For distinct users, set `APP_AUTH_MODE=individual` and configure
`APP_PRINCIPAL_API_KEYS` as a JSON mapping from stable principal IDs to distinct
secret keys. The server derives the principal from the verified key; a caller
cannot choose it with a header, body field, IP address, or thread ID. Empty or
duplicate credentials fail closed with `503`. The principal is bound to the
conversation before checkpoint lookup; cross-principal reuse gets a generic
`404` without model or summarizer calls. Expiry and eviction remove both the
ownership record and checkpoint. Repository selection remains a separate check;
the curated catalog is visible to every authenticated principal and contains no
per-user private repositories.

Keep API keys on a trusted backend. A browser should authenticate to that backend
using its own secure session; the backend maps that session to one configured
principal credential and forwards `/ask` and `/repos` server to server. Never put
`APP_API_KEY` or an individual key in JavaScript, a public environment variable,
or a browser request. Rotating a key for the same principal preserves ownership
while the process retains the conversation. Changing the principal ID creates a
new identity. This key map is suitable for a small controlled deployment; it has
no self-service account provisioning or OAuth.

## Usage Controls

Authenticated `/ask` requests are admitted before the graph starts. The process
tracks accepted requests in a sliding window per client IP and globally
(`APP_ASK_RATE_PER_CLIENT`, `APP_ASK_RATE_GLOBAL`,
`APP_ASK_RATE_WINDOW_SECONDS`) and active requests with separate per-client and
global concurrency limits. A rejected request returns `429` with `Retry-After`
and does not call the model. Capacity is released after success or error. The
lease spans summarization, chat, and semantic-search embeddings.

The client identity is the socket peer IP, without trusting forwarded headers.
Users behind one proxy may share a quota. Counters are in process memory, so
restarts reset them and multiple workers or instances each enforce their own
limits. These controls reduce accidental or local abuse but are not a shared
deployment-wide financial quota.

`APP_PROVIDER_TIMEOUT_SECONDS` and `APP_PROVIDER_MAX_RETRIES` configure chat and
embedding clients. The remaining `APP_ASK_DEADLINE_SECONDS` is checked before
each provider call and caps its configured timeout. The same policy applies to
summarization. `APP_MODEL_MAX_INPUT_CHARS` rejects oversized chat input and
skips oversized summaries or embedding batches; chat responses request at most
`APP_MODEL_MAX_COMPLETION_TOKENS`. A client-side timeout or deadline does not
guarantee cancellation of synchronous work already running in a provider,
library, or tool. A retry setting above zero can also extend a single provider
call; the default is zero. `APP_MAX_ITERATIONS` only counts tool calls, so it is
not an exact token, time, or cost ceiling. Aggregate token and cost measurement
remain separate work.

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

## Prompt-injection boundary

Repository files and tool responses are untrusted input. The agent receives
them as tool messages, while the trusted system message tells it to treat
instructions found there as source data. A rolling conversation summary may
contain text derived from those files, so it is supplied in a separate,
labeled historical-context message, outside the system message. The
summarizer is instructed to preserve provenance and avoid adopting embedded
instructions. File eligibility and the per-turn tool budget remain enforced
in code regardless of model behavior.

`tests/test_prompt_injection.py` checks message placement and malicious
fixtures offline. `uv run python -m eval.run --adversarial` is an opt-in real
model evaluation that records task deviation, unauthorized access attempts,
and later-turn contamination. Prompt wording and these finite tests cannot
guarantee immunity; the rubric can also misclassify answers that merely
discuss suspicious text.

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

`/metrics` requires the same API key as `/ask`. Its private registry contains
only application request counts and duration/tool-call histograms. Outcomes are
mapped to a fixed label set; questions, thread IDs, repo paths, and error text
are not metric labels. Metrics are held in process memory and reset on restart.

Tracing through LangSmith is optional and disabled by default. Enabling it may
send prompts, repository tool output, and responses to an external service;
use it only in an environment where that data may be shared.

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
| Static shared API key | Study mode has one principal; individual mode needs a trusted server to hold each user's key. No rotation API. |
| Process-local rate limits | A leaked valid key can still spend tokens within each process's limits; no shared quota across instances. |
| Curated YAML trust boundary | `git_url` values are trusted configuration, not user input. |
| Diagnostic logs may include sensitive content | Enable only for controlled troubleshooting; private mode is the default. |
| Conversation memory and summaries in process | No durable store, no encryption-at-rest concerns inside this app, but no persistence guarantees. |
| Semantic search sends file content to embedding provider | Only eligible non-sensitive files are embedded, but repo content still leaves the process when the feature is enabled. |
| Repository content exposure | Tools expose non-sensitive text files from configured repos to the LLM and response trajectory summaries. |

## Not Implemented

- OAuth or self-service user authentication.
- Authorization by repo; the curated catalog is shared by all authenticated principals.
- Shared rate limiting or quotas across instances.
- SSRF allowlist for caller-submitted URLs, because caller-submitted URLs are not supported.
- Secret scanning beyond filename pattern filtering.
- A deployed metrics store or default tracing backend.
