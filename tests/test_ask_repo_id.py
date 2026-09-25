from langchain_core.messages import AIMessage
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

from app.main import compiled_graph

from tests.test_graph import FakeTool
from tests.test_health import FakeReActLLM


def _tool_call_message():
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"relative_path": "README.md"}, "id": "call_1"}
        ],
    )


def test_ask_with_valid_repo_id_uses_its_repo_path(client, monkeypatch):
    fake_tool = FakeTool(result="file contents")
    monkeypatch.setattr("app.graph.get_tool_registry", lambda: {"read_file": fake_tool})
    monkeypatch.setattr(
        "app.main.repo_registry", {"other-project": "/path/to/other-project"}
    )
    fake_llm = FakeReActLLM([_tool_call_message(), AIMessage(content="done")])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    response = client.post(
        "/ask", json={"question": "What's in README.md?", "repo_id": "other-project"}
    )

    assert response.status_code == 200
    assert fake_tool.last_args["repo_path"] == "/path/to/other-project"


def test_ask_with_unknown_repo_id_returns_404(client):
    response = client.post(
        "/ask", json={"question": "What's in README.md?", "repo_id": "does-not-exist"}
    )

    assert response.status_code == 404


def test_ask_without_repo_id_falls_back_to_default_repo_path(client, monkeypatch):
    fake_tool = FakeTool(result="file contents")
    monkeypatch.setattr("app.graph.get_tool_registry", lambda: {"read_file": fake_tool})
    monkeypatch.setattr("app.main.settings.repo_path", "/default/repo/path")
    fake_llm = FakeReActLLM([_tool_call_message(), AIMessage(content="done")])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    response = client.post("/ask", json={"question": "What's in README.md?"})

    assert response.status_code == 200
    assert fake_tool.last_args["repo_path"] == "/default/repo/path"


def test_thread_rejects_repo_change_before_llm_or_state_update(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.repo_path", "/default/repo")
    monkeypatch.setattr("app.main.settings.max_history_messages", 1)
    monkeypatch.setattr("app.main.repo_registry", {"other": "/other/repo"})
    fake_llm = FakeReActLLM([AIMessage(content="first answer"), AIMessage(content="follow up")])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    def unexpected_summary(*args):
        raise AssertionError("Rejected request must not summarize")

    monkeypatch.setattr("app.main.build_conversation_summary", unexpected_summary)

    first = client.post("/ask", json={"question": "First question", "thread_id": "bound-1"})
    config = {"configurable": {"thread_id": "bound-1"}}
    before = compiled_graph.get_state(config).values.copy()
    changed = client.post(
        "/ask", json={"question": "Other project?", "thread_id": "bound-1", "repo_id": "other"}
    )

    assert first.status_code == 200
    assert changed.status_code == 409
    assert "new" in changed.json()["detail"].lower()
    assert fake_llm.invocations == 1
    assert compiled_graph.get_state(config).values == before

    monkeypatch.setattr("app.main.settings.max_history_messages", 100)
    resumed = client.post("/ask", json={"question": "Same project?", "thread_id": "bound-1"})
    assert resumed.status_code == 200
    assert resumed.json()["answer"] == "follow up"


def test_default_repo_and_catalog_alias_share_thread(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.repo_path", "/default/repo")
    monkeypatch.setattr("app.main.repo_registry", {"alias": "/default/repo/."})
    fake_llm = FakeReActLLM([AIMessage(content="first"), AIMessage(content="second")])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    first = client.post("/ask", json={"question": "First question", "thread_id": "alias-1"})
    second = client.post(
        "/ask", json={"question": "Follow up", "thread_id": "alias-1", "repo_id": "alias"}
    )

    assert first.status_code == second.status_code == 200
    assert fake_llm.invocations == 2


def test_simultaneous_first_requests_cannot_bind_different_repos(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.repo_path", "/default/repo")
    monkeypatch.setattr("app.main.repo_registry", {"other": "/other/repo"})
    fake_llm = FakeReActLLM([AIMessage(content="first")])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)
    original_invoke = compiled_graph.invoke
    entered = Event()
    release = Event()
    second_entered = Event()
    calls_lock = Lock()
    calls = 0

    def blocking_invoke(*args, **kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            entered.set()
            assert release.wait(5)
        else:
            second_entered.set()
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(compiled_graph, "invoke", blocking_invoke)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            client.post, "/ask", json={"question": "First question", "thread_id": "race-1"}
        )
        assert entered.wait(5)
        second = pool.submit(
            client.post,
            "/ask",
            json={"question": "Other project?", "thread_id": "race-1", "repo_id": "other"},
        )
        try:
            assert not second_entered.wait(0.2)
        finally:
            release.set()
        responses = [first.result(timeout=5), second.result(timeout=5)]

    assert sorted(response.status_code for response in responses) == [200, 409]
    assert fake_llm.invocations == 1
