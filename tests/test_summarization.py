from langchain_core.messages import AIMessage, HumanMessage

from app.summarization import build_conversation_summary


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
