"""Continuation tests with a fake model that enforces tool-calling pairing.

Providers that require pairing reject a prompt where a tool call has no
result or a tool result has no matching call. The fake below enforces those
rules on every prompt, so these tests fail wherever history compaction or the
budget guardrail leaves the persisted conversation unpaired.
"""

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.i18n import BUDGET_EXCEEDED_MESSAGES
from app.main import compiled_graph


def assert_valid_tool_protocol(messages):
    assert isinstance(messages[0], SystemMessage), "prompt must start with system"
    pending: set[str] = set()
    for index, message in enumerate(messages[1:], start=1):
        assert not isinstance(message, SystemMessage), f"system message at {index}"
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, (
                f"orphan tool result {message.tool_call_id} at {index}"
            )
            pending.remove(message.tool_call_id)
            continue
        assert not pending, f"unanswered tool calls {sorted(pending)} before {index}"
        if isinstance(message, AIMessage) and message.tool_calls:
            pending = {tool_call["id"] for tool_call in message.tool_calls}
    assert not pending, f"unanswered tool calls {sorted(pending)} at end of prompt"


class ProtocolValidatingLLM:
    """Scripted agent model plus a summarizer that may be told to fail."""

    def __init__(self, responses, summary="summary", summary_error=None):
        self._responses = list(responses)
        self._summary = summary
        self._summary_error = summary_error
        self.prompts = []
        self.summary_transcripts = []

    def bind_tools(self, tools):
        return _BoundProtocolLLM(self)

    def invoke(self, messages):
        self.summary_transcripts.append(messages[-1].content)
        if self._summary_error is not None:
            raise self._summary_error
        return AIMessage(content=self._summary)


class _BoundProtocolLLM:
    def __init__(self, parent):
        self._parent = parent

    def invoke(self, messages):
        assert_valid_tool_protocol(messages)
        self._parent.prompts.append(list(messages))
        if not self._parent._responses:
            raise AssertionError("Fake LLM received more invocations than expected")
        return self._parent._responses.pop(0)


class CountingTool:
    def __init__(self, name, error=None):
        self.name = name
        self._error = error
        self.invocations = 0

    def invoke(self, args):
        self.invocations += 1
        if self._error is not None:
            raise self._error
        return f"{self.name} output"


def _tool_call(name, call_id, args=None):
    return {"name": name, "args": args or {}, "id": call_id}


def _install(monkeypatch, fake_llm, tools):
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)
    monkeypatch.setattr(
        "app.graph.get_tool_registry", lambda: {tool.name: tool for tool in tools}
    )


def _stored_messages(thread_id):
    state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    return state.values["messages"]


def _ask(client, question, thread_id):
    return client.post("/ask", json={"question": question, "thread_id": thread_id})


def test_budget_exceeded_turn_is_closed_and_next_turn_is_valid(client, monkeypatch):
    monkeypatch.setattr("app.graph.settings.max_iterations", 1)
    read_file = CountingTool("read_file")
    list_files = CountingTool("list_files")
    fake_llm = ProtocolValidatingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("read_file", "call_1", {"relative_path": "a.py"}),
                    _tool_call("list_files", "call_2"),
                ],
            ),
            AIMessage(content="", tool_calls=[_tool_call("list_files", "call_3")]),
            AIMessage(content="answer after budget reset"),
        ]
    )
    _install(monkeypatch, fake_llm, [read_file, list_files])

    first = _ask(client, "Explain everything", "budget-protocol")
    assert first.status_code == 200
    assert first.json()["answer"] == BUDGET_EXCEEDED_MESSAGES["pt-BR"]
    assert first.json()["iterations"] == 0
    assert read_file.invocations == list_files.invocations == 0

    stored = _stored_messages("budget-protocol")
    rejected = [message for message in stored if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in rejected] == ["call_1", "call_2"]
    assert all(message.status == "error" for message in rejected)
    assert isinstance(stored[-1], AIMessage)
    assert stored[-1].content == BUDGET_EXCEEDED_MESSAGES["pt-BR"]
    assert not stored[-1].tool_calls

    second = _ask(client, "Just list the files", "budget-protocol")
    assert second.status_code == 200
    assert second.json()["answer"] == "answer after budget reset"
    assert second.json()["iterations"] == 1
    assert list_files.invocations == 1
    assert read_file.invocations == 0
    assert [step["tool"] for step in second.json()["trajectory"]] == [
        "list_files",
        "agent_decide",
    ]
    assert fake_llm.prompts[-1][-1].content == "list_files output"


def _multi_tool_turn_responses():
    return [
        AIMessage(
            content="",
            tool_calls=[
                _tool_call("read_file", "call_1", {"relative_path": "src/main.py"}),
                _tool_call("list_files", "call_2"),
            ],
        ),
        AIMessage(
            content="", tool_calls=[_tool_call("grep_repo", "call_3", {"term": "x"})]
        ),
        AIMessage(content="answer 1"),
        AIMessage(content="answer 2"),
        AIMessage(content="answer 3"),
    ]


def _multi_tool_tools():
    return [
        CountingTool("read_file"),
        CountingTool("list_files"),
        CountingTool("grep_repo"),
    ]


# Turn 1 stores 7 messages (human, ai+2 calls, 2 results, ai+1 call, result,
# answer) and turn 2 adds 2 more, so limits 1..8 put the compaction cut before
# turn 3 at every position of the 9-message history.
@pytest.mark.parametrize("max_history", range(1, 9))
def test_compaction_never_splits_tool_call_groups(client, monkeypatch, max_history):
    monkeypatch.setattr("app.main.settings.max_history_messages", max_history)
    fake_llm = ProtocolValidatingLLM(_multi_tool_turn_responses())
    _install(monkeypatch, fake_llm, _multi_tool_tools())
    thread_id = f"compact-{max_history}"

    for turn in range(1, 4):
        response = _ask(client, f"question {turn}", thread_id)
        assert response.status_code == 200
        assert response.json()["answer"] == f"answer {turn}"

    stored = _stored_messages(thread_id)
    assert stored[0].content in {"question 2", "question 3"}
    assert stored[-2].content == "question 3"
    assert fake_llm.summary_transcripts


def test_summary_keeps_tool_names_arguments_ids_and_evidence(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_history_messages", 8)
    fake_llm = ProtocolValidatingLLM(_multi_tool_turn_responses())
    _install(monkeypatch, fake_llm, _multi_tool_tools())

    for turn in range(1, 4):
        assert _ask(client, f"question {turn}", "summary-tools").status_code == 200

    transcript = fake_llm.summary_transcripts[-1]
    assert 'call_1: read_file({"relative_path": "src/main.py"})' in transcript
    assert "Tool result for call_1 (read_file): read_file output" in transcript
    assert 'call_3: grep_repo({"term": "x"})' in transcript
    assert "Tool result for call_3 (grep_repo): grep_repo output" in transcript


@pytest.mark.parametrize("max_history", range(1, 9))
def test_summarization_failure_still_compacts_complete_turns(
    client, monkeypatch, max_history
):
    monkeypatch.setattr("app.main.settings.max_history_messages", max_history)
    fake_llm = ProtocolValidatingLLM(
        _multi_tool_turn_responses(), summary_error=RuntimeError("llm unavailable")
    )
    _install(monkeypatch, fake_llm, _multi_tool_tools())
    thread_id = f"summary-fail-{max_history}"

    for turn in range(1, 4):
        response = _ask(client, f"question {turn}", thread_id)
        assert response.status_code == 200
        assert response.json()["answer"] == f"answer {turn}"

    state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    assert state.values.get("conversation_summary", "") == ""
    assert state.values["messages"][0].content in {"question 2", "question 3"}


def test_handled_tool_error_is_paired_and_next_turn_is_valid(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_history_messages", 3)
    failing = CountingTool("read_file", error=FileNotFoundError("missing.py"))
    fake_llm = ProtocolValidatingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("read_file", "call_1", {"relative_path": "missing.py"})
                ],
            ),
            AIMessage(content="the file does not exist"),
            AIMessage(content="follow-up answer"),
        ]
    )
    _install(monkeypatch, fake_llm, [failing])

    first = _ask(client, "Read missing.py", "tool-error")
    second = _ask(client, "And now?", "tool-error")

    assert first.status_code == second.status_code == 200
    assert first.json()["answer"] == "the file does not exist"
    assert second.json()["answer"] == "follow-up answer"
    assert "Tool error: missing.py" in fake_llm.summary_transcripts[-1]


def test_unexpected_tool_crash_does_not_leave_pending_calls(client, monkeypatch):
    crashing = CountingTool("read_file", error=RuntimeError("tool crashed"))
    fake_llm = ProtocolValidatingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("read_file", "call_1", {"relative_path": "a.py"}),
                    _tool_call("read_file", "call_2", {"relative_path": "b.py"}),
                ],
            ),
            AIMessage(content="recovered answer"),
        ]
    )
    _install(monkeypatch, fake_llm, [crashing])

    first = _ask(client, "Read a.py and b.py", "tool-crash")
    assert first.status_code == 500

    second = _ask(client, "Try something else", "tool-crash")
    assert second.status_code == 200
    assert second.json()["answer"] == "recovered answer"
    closed = [
        message
        for message in fake_llm.prompts[-1]
        if isinstance(message, ToolMessage)
    ]
    assert [message.tool_call_id for message in closed] == ["call_1", "call_2"]
    assert all(message.status == "error" for message in closed)
