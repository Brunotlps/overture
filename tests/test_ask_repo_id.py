from langchain_core.messages import AIMessage

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
