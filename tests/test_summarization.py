from langchain_core.messages import AIMessage, HumanMessage

from app.summarization import SUMMARY_MAX_CHARS, build_conversation_summary


def test_summarizes_dropped_messages_using_provided_summarize_fn():
    messages = [
        HumanMessage(content="What does this service do?"),
        AIMessage(content="It answers questions about a git repo."),
    ]
    calls = []

    def fake_summarize_fn(transcript):
        calls.append(transcript)
        return "Summary of the conversation so far."

    summary = build_conversation_summary(messages, "", fake_summarize_fn)

    assert summary == "Summary of the conversation so far."
    assert len(calls) == 1
    assert "What does this service do?" in calls[0]
    assert "It answers questions about a git repo." in calls[0]


def test_combines_prior_summary_with_newly_dropped_messages():
    messages = [HumanMessage(content="And what about /repos?")]
    calls = []

    def fake_summarize_fn(transcript):
        calls.append(transcript)
        return "Updated summary."

    summary = build_conversation_summary(
        messages, "User asked what the service does.", fake_summarize_fn
    )

    assert summary == "Updated summary."
    assert "User asked what the service does." in calls[0]
    assert "And what about /repos?" in calls[0]


def test_summary_is_truncated_to_max_chars():
    messages = [HumanMessage(content="question")]

    def fake_summarize_fn(transcript):
        return "x" * (SUMMARY_MAX_CHARS + 500)

    summary = build_conversation_summary(messages, "", fake_summarize_fn)

    assert len(summary) == SUMMARY_MAX_CHARS
