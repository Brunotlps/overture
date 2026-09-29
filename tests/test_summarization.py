from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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


def test_transcript_pairs_tool_calls_with_their_results():
    messages = [
        HumanMessage(content="Where is /ask defined?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "grep_repo", "args": {"term": "/ask"}, "id": "call_1"},
                {"name": "list_files", "args": {}, "id": "call_2"},
            ],
        ),
        ToolMessage(content='app/main.py:103: @app.post("/ask")', tool_call_id="call_1"),
        ToolMessage(content="Tool error: boom", tool_call_id="call_2"),
        AIMessage(content="It is defined in app/main.py."),
    ]
    calls = []

    def fake_summarize_fn(transcript):
        calls.append(transcript)
        return "summary"

    build_conversation_summary(messages, "", fake_summarize_fn)

    assert calls[0].splitlines() == [
        "Human: Where is /ask defined?",
        'AI tool call call_1: grep_repo({"term": "/ask"})',
        "AI tool call call_2: list_files({})",
        'Tool result for call_1 (grep_repo): app/main.py:103: @app.post("/ask")',
        "Tool result for call_2 (list_files): Tool error: boom",
        "AI: It is defined in app/main.py.",
    ]
