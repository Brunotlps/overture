from typing import Callable

from langchain_core.messages import BaseMessage

SummarizeFn = Callable[[str], str]


def _format_transcript(messages: list[BaseMessage], prior_summary: str) -> str:
    lines = []
    if prior_summary:
        lines.append(f"Summary so far: {prior_summary}")
    for message in messages:
        role = message.__class__.__name__.removesuffix("Message")
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def build_conversation_summary(
    messages: list[BaseMessage], prior_summary: str, summarize_fn: SummarizeFn
) -> str:
    """Summarize messages being dropped from history, folding in any prior summary."""
    transcript = _format_transcript(messages, prior_summary)
    return summarize_fn(transcript)
