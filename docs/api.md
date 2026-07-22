# API

Overture exposes three HTTP endpoints from `app.main`.

## Authentication

`/ask` and `/repos` require:

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

If `portfolio_repos.yaml` is absent or all configured repos fail to materialize,
the endpoint returns an empty list.

Source: `app.main.list_repos`, `app.schemas.RepoInfo`.

## `POST /ask`

Asks a question about the default repository or a curated portfolio repository.

Requires `X-API-Key`.

Request:

```json
{
  "question": "How does /ask work?",
  "thread_id": "optional-conversation-id",
  "repo_id": "optional-curated-repo-id"
}
```

Fields:

| Field | Required | Constraints | Behavior |
| --- | --- | --- | --- |
| `question` | yes | 3 to 500 characters | Natural-language question sent to the agent. |
| `thread_id` | no | max 100 characters | Reuse to continue an in-memory conversation. Omit for a fresh thread. |
| `repo_id` | no | must exist in startup registry | Selects a curated repo. Omit to use `APP_REPO_PATH`. |

Unknown fields are rejected with `422` because `AskRequest` uses `extra="forbid"`.
The legacy `target` request field is no longer accepted.

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
| `422` | Request body failed Pydantic validation. |
| `500` | Unexpected graph/runtime failure; response detail is intentionally generic. |

## Trajectory

`trajectory` records the graph-visible steps taken to answer. Common `tool` values:

- `agent_decide` - the LLM produced the final answer or an empty-answer fallback.
- `list_files` - the LLM requested repository file listing.
- `read_file` - the LLM requested a file read.
- `grep_repo` - the LLM requested exact substring search.
- `semantic_search` - the LLM requested meaning-based file lookup. Present only when `APP_SEMANTIC_SEARCH_ENABLED=true`.
- `max_iterations_guardrail` - a requested tool batch exceeded the remaining budget.

`repo_path` is injected internally into tools and is not included in `tool_input`.

## Conversation Memory

When `thread_id` is reused, LangGraph `MemorySaver` provides prior conversation
messages to the graph. This memory is process-local only. It does not survive
application restarts or Fly scale-to-zero.

When a thread exceeds `APP_MAX_HISTORY_MESSAGES`, the oldest messages are removed
from message history and folded into a rolling `conversation_summary`. That summary
is injected into the system prompt on later turns. If the summarization LLM call
fails, `/ask` continues by dropping those messages without updating the summary.
