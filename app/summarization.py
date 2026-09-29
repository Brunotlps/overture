import json
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

SummarizeFn = Callable[[str], str]

SUMMARY_MAX_CHARS = 1000


def _format_transcript(messages: list[BaseMessage], prior_summary: str) -> str:
    lines = []
    if prior_summary:
        lines.append(f"Summary so far: {prior_summary}")
    tool_names: dict[str, str] = {}
    for message in messages:
        role = message.__class__.__name__.removesuffix("Message")
        if isinstance(message, ToolMessage):
            tool_name = message.name or tool_names.get(message.tool_call_id, "unknown")
            lines.append(
                f"Tool result for {message.tool_call_id} ({tool_name}): "
                f"{message.content}"
            )
            continue
        if message.content or not isinstance(message, AIMessage):
            lines.append(f"{role}: {message.content}")
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_names[tool_call["id"]] = tool_call["name"]
                args = json.dumps(tool_call["args"], sort_keys=True)
                lines.append(
                    f"AI tool call {tool_call['id']}: {tool_call['name']}({args})"
                )
    return "\n".join(lines)


def build_conversation_summary(
    messages: list[BaseMessage],
    prior_summary: str,
    summarize_fn: SummarizeFn,
    max_chars: int = SUMMARY_MAX_CHARS,
) -> str:
    """Summarize messages being dropped from history, folding in any prior summary.

    Defensively truncated to max_chars in case summarize_fn's own conciseness
    instruction is ignored by the model, so the summary can't grow unbounded.
    """
    transcript = _format_transcript(messages, prior_summary)
    return summarize_fn(transcript)[:max_chars]
