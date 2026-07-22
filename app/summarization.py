from typing import Callable

from langchain_core.messages import BaseMessage

SummarizeFn = Callable[[str], str]

SUMMARY_MAX_CHARS = 1000


def _format_transcript(messages: list[BaseMessage], prior_summary: str) -> str:
    lines = []
    if prior_summary:
        lines.append(f"Summary so far: {prior_summary}")
    for message in messages:
        role = message.__class__.__name__.removesuffix("Message")
        lines.append(f"{role}: {message.content}")
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
