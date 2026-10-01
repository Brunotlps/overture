# API

Overture exposes four HTTP endpoints from `app.main`.

## Authentication

`/ask`, `/repos`, and `/metrics` require:

```http
X-API-Key: <APP_API_KEY>
```

If `APP_API_KEY` is unset on the server, authenticated endpoints return `503`.
If the header is missing or wrong, they return `401`. `/health` is public.

## `GET /health`

Returns service health and application version.

Response:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

Source: `app.main.health`.

## `GET /metrics`

Returns Prometheus text exposition for process-local `/ask` metrics. Requires
`X-API-Key`; `/health` remains public. The endpoint exposes a fixed set of
`outcome` labels and never includes questions, thread IDs, repo paths, or API
keys. See [Operations](operations.md#metrics-and-tracing) for metric names and
queries. Each process has independent counters that reset on restart.

## `GET /repos`

Lists curated portfolio repositories that were successfully registered at startup.

Requires `X-API-Key`.

Response:

```json
[
  {
    "repo_id": "overture",
    "display_name": "Overture"
  }
]
```

The repository includes a default `portfolio_repos.yaml` with `overture`, `codda`,
`briskmail`, and `interlude`. If that file is absent in another deployment or all
configured repos fail to materialize, the endpoint returns an empty list.

Source: `app.main.list_repos`, `app.schemas.RepoInfo`.

## `POST /ask`

Asks a question about the default repository or a curated portfolio repository.

Requires `X-API-Key`.

Request:

```json
{
  "question": "How does /ask work?",
  "thread_id": "optional-conversation-id",
  "repo_id": "optional-curated-repo-id",
  "language": "pt-BR"
}
```

Fields:

| Field | Required | Constraints | Behavior |
| --- | --- | --- | --- |
| `question` | yes | 3 to 500 characters | Natural-language question sent to the agent. |
| `thread_id` | no | max 100 characters | Reuse to continue an in-memory conversation. Omit for a fresh thread. |
| `repo_id` | no | must exist in startup registry | Selects a curated repo. Omit to use `APP_REPO_PATH`. |
| `language` | no | `pt-BR` or `en` | Selects the answer language. Defaults to `pt-BR`. |

Unknown fields are rejected with `422` because `AskRequest` uses `extra="forbid"`.
The legacy `target` request field is no longer accepted.

`language` is per-request. Reusing a `thread_id` does not pin the conversation to
one language; switching `language` mid-thread changes the answer language for that
request. Code identifiers, file paths, and quoted code are kept in their original
form. Internal prompts, logs, and HTTP error `detail` strings remain in English.

Response:

```json
{
  "answer": "The agent's final answer.",
  "trajectory": [
    {
      "tool": "read_file",
      "tool_input": "{\"relative_path\": \"app/main.py\"}",
      "output_summary": "executed successfully"
    }
  ],
  "iterations": 1,
  "thread_id": "conversation-id"
}
```

Status codes:

| Status | Meaning |
| --- | --- |
| `200` | Agent completed and returned an answer or guardrail message. |
| `401` | Missing or invalid API key. |
| `404` | `repo_id` was provided but is unknown. |
| `409` | The thread already belongs to another repository. Start a new conversation to switch projects. |
| `413` | The assembled chat input exceeded `APP_MODEL_MAX_INPUT_CHARS`. |
| `429` | Per-client or global `/ask` rate/concurrency quota exceeded. Includes `Retry-After` in seconds; no graph or model call starts. |
| `422` | Request body failed Pydantic validation. |
| `500` | Unexpected graph/runtime failure; response detail is intentionally generic. |
| `504` | The model request timed out or the request deadline was exhausted before another provider call. |

## Trajectory

`trajectory` records the graph-visible steps taken to answer. Common `tool` values:

- `agent_decide` - the LLM produced the final answer or an empty-answer fallback.
- `list_files` - the LLM requested repository file listing.
- `read_file` - the LLM requested a file read.
- `grep_repo` - the LLM requested exact substring search.
- `semantic_search` - the LLM requested meaning-based file lookup. Present only when `APP_SEMANTIC_SEARCH_ENABLED=true`.
- `max_iterations_guardrail` - a requested tool batch exceeded the remaining budget.

`repo_path` is injected internally into tools and is not included in `tool_input`.
Expected tool failures keep the same trajectory shape but use stable error
codes in `output_summary`; raw exception text is not returned. `tool_input`
still shows the selected arguments, capped at 2,000 characters for a tool call.

## Conversation Memory

When `thread_id` is reused, the in-memory LangGraph checkpointer provides prior
conversation messages to the graph. This memory is process-local only. It does not
survive application restarts or Fly scale-to-zero.

Requests without `thread_id` get a new one that can be reused like any other.
Retention is bounded: each thread keeps only its latest checkpoint, and whole
threads are deleted after `APP_THREAD_TTL_SECONDS` (default 24 hours) without use
or, beyond `APP_MAX_THREADS` (default 500), least recently used first. A thread
with a request in flight is never deleted. Reusing an expired or evicted
`thread_id` is not an error: it starts a fresh conversation under the same ID,
without prior messages, summary, or repository binding.
The first request binds a thread to the resolved repository path. A later
request for another repository returns `409` before summarization or agent
execution. Omitting `repo_id` and selecting a catalog alias for the same path
are equivalent. Requests for the same thread are serialized within a process.

When a thread exceeds `APP_MAX_HISTORY_MESSAGES`, the oldest whole turns are removed
(so a tool call is never separated from its results) from message history and folded into a rolling `conversation_summary`. That summary
is supplied as labeled, untrusted context outside the system message on later turns. If the summarization LLM call
fails, `/ask` continues by dropping those messages without updating the summary.
