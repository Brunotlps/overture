from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import main
from app.config import settings
from app.graph import build_react_graph
from app.retention import LatestCheckpointSaver, ThreadRetention
from tests.test_health import FakeReActLLM
from tests.test_retention import FakeClock


def test_principal_cannot_resume_another_conversation_before_checkpoint_or_llm(
    monkeypatch,
):
    saver = LatestCheckpointSaver()
    graph = build_react_graph(checkpointer=saver)
    monkeypatch.setattr(main, "compiled_graph", graph)
    monkeypatch.setattr(main, "thread_retention", ThreadRetention(saver))
    monkeypatch.setattr(settings, "auth_mode", "individual", raising=False)
    monkeypatch.setattr(
        settings,
        "principal_api_keys",
        {"alice": "alice-secret", "bob": "bob-secret"},
        raising=False,
    )
    llm = FakeReActLLM([AIMessage(content="private answer")])
    monkeypatch.setattr("app.graph.get_llm", lambda: llm)
    alice = TestClient(main.app, headers={"X-API-Key": "alice-secret"})
    bob = TestClient(main.app, headers={"X-API-Key": "bob-secret"})

    created = alice.post(
        "/ask", json={"question": "Private question", "thread_id": "known-thread"}
    )
    assert created.status_code == 200

    with patch.object(graph, "get_state", side_effect=AssertionError("checkpoint read")):
        denied = bob.post(
            "/ask", json={"question": "Steal context", "thread_id": "known-thread"}
        )

    assert denied.status_code == 404
    assert "private" not in denied.text.lower()
    assert llm.invocations == 1


def test_individual_mode_rejects_shared_service_key(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "individual", raising=False)
    monkeypatch.setattr(
        settings, "principal_api_keys", {"alice": "alice-secret"}, raising=False
    )
    response = TestClient(main.app, headers={"X-API-Key": "shared-key"}).post(
        "/ask", json={"question": "Hello there"}
    )
    assert response.status_code == 401


def test_expiration_removes_owner_and_checkpoint_before_new_principal_claims_id(
    monkeypatch,
):
    saver = LatestCheckpointSaver()
    clock = FakeClock()
    graph = build_react_graph(checkpointer=saver)
    monkeypatch.setattr(main, "compiled_graph", graph)
    monkeypatch.setattr(main, "thread_retention", ThreadRetention(saver, clock=clock))
    monkeypatch.setattr(settings, "auth_mode", "individual")
    monkeypatch.setattr(settings, "principal_api_keys", {"alice": "a", "bob": "b"})
    monkeypatch.setattr(settings, "thread_ttl_seconds", 10)
    llm = FakeReActLLM([AIMessage(content="alice reply"), AIMessage(content="bob reply")])
    monkeypatch.setattr("app.graph.get_llm", lambda: llm)

    alice = TestClient(main.app, headers={"X-API-Key": "a"})
    bob = TestClient(main.app, headers={"X-API-Key": "b"})
    assert alice.post(
        "/ask", json={"question": "Alice question", "thread_id": "reused"}
    ).status_code == 200
    clock.now += 11
    resumed = bob.post(
        "/ask", json={"question": "Bob question", "thread_id": "reused"}
    )
    assert resumed.status_code == 200
    assert resumed.json()["answer"] == "bob reply"
    messages = graph.get_state({"configurable": {"thread_id": "reused"}}).values[
        "messages"
    ]
    assert "Alice question" not in [message.content for message in messages]
    assert alice.post(
        "/ask", json={"question": "Try again", "thread_id": "reused"}
    ).status_code == 404


def test_configured_principal_keys_must_be_unique(monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "individual")
    monkeypatch.setattr(settings, "principal_api_keys", {"alice": "same", "bob": "same"})
    response = TestClient(main.app, headers={"X-API-Key": "same"}).post(
        "/ask", json={"question": "Hello there"}
    )
    assert response.status_code == 503


def test_eviction_deletes_owner_binding(monkeypatch):
    saver = LatestCheckpointSaver()
    retention = ThreadRetention(saver)
    monkeypatch.setattr(settings, "max_threads", 1)
    retention.begin("old", "alice")
    retention.end("old")
    retention.begin("new", "alice")
    retention.end("new")

    assert retention.thread_ids() == ["new"]
    retention.begin("old", "bob")
    retention.end("old")
    assert retention.thread_ids() == ["old"]
