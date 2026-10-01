"""Opt-in PostgreSQL integration tests; run with OVERTURE_TEST_POSTGRES_DSN."""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import psycopg
import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

from app import main
from app.config import Settings
from app.retention import StorageUnavailable
from app.schemas import AskRequest
from tests.test_health import FakeReActLLM


@pytest.fixture
def postgres_dsn():
    dsn = os.getenv("OVERTURE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set OVERTURE_TEST_POSTGRES_DSN for PostgreSQL integration")
    return dsn


def test_conversation_survives_graph_reconstruction_and_rejects_other_principal(
    postgres_dsn, monkeypatch
):
    from app.persistence import open_postgres_runtime

    thread_id = f"rebuild-{uuid.uuid4().hex}"
    monkeypatch.setattr(main.settings, "checkpointer_backend", "postgres")
    llm = FakeReActLLM([AIMessage(content="first"), AIMessage(content="second")])
    monkeypatch.setattr("app.graph.get_llm", lambda: llm)
    request = AskRequest(question="First question", thread_id=thread_id)

    with open_postgres_runtime(postgres_dsn, setup=True) as runtime:
        monkeypatch.setattr(main, "compiled_graph", runtime.graph)
        monkeypatch.setattr(main, "thread_retention", runtime.retention)
        first = main.ask(request, principal_id="individual:alice")
        assert first.answer == "first"

    with open_postgres_runtime(postgres_dsn) as rebuilt:
        monkeypatch.setattr(main, "compiled_graph", rebuilt.graph)
        monkeypatch.setattr(main, "thread_retention", rebuilt.retention)
        with pytest.raises(HTTPException) as denied:
            main.ask(request, principal_id="individual:bob")
        assert denied.value.status_code == 404
        second = main.ask(
            AskRequest(question="Follow up", thread_id=thread_id),
            principal_id="individual:alice",
        )
        assert second.answer == "second"
        messages = rebuilt.graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        ).values["messages"]
        assert "First question" in [message.content for message in messages]


def test_two_instances_serialize_same_thread(postgres_dsn):
    from app.persistence import open_postgres_runtime

    entered = Event()
    release = Event()
    second_entered = Event()

    thread_id = f"cross-instance-{uuid.uuid4().hex}"
    with (
        open_postgres_runtime(postgres_dsn, setup=True) as first,
        open_postgres_runtime(postgres_dsn) as second,
    ):
        def hold_first():
            first.retention.begin(thread_id, "individual:alice", "/repo")
            entered.set()
            assert release.wait(5)
            first.retention.end(thread_id)

        def start_second():
            assert entered.wait(5)
            second.retention.begin(thread_id, "individual:alice", "/repo")
            second_entered.set()
            second.retention.end(thread_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(hold_first)
            two = pool.submit(start_second)
            assert entered.wait(5)
            try:
                assert not second_entered.wait(0.2)
            finally:
                release.set()
            one.result(timeout=5)
            two.result(timeout=5)
        assert second_entered.is_set()


def test_owner_repo_and_expiry_are_bound_to_durable_checkpoint(
    postgres_dsn, monkeypatch
):
    from app.persistence import open_postgres_runtime

    thread_id = f"expiry-{uuid.uuid4().hex}"
    monkeypatch.setattr(main.settings, "checkpointer_backend", "postgres")
    monkeypatch.setattr(main.settings, "repo_path", "/default/repo")
    monkeypatch.setitem(main.repo_registry, "other", "/other/repo")
    llm = FakeReActLLM([AIMessage(content="first"), AIMessage(content="new")])
    monkeypatch.setattr("app.graph.get_llm", lambda: llm)
    with open_postgres_runtime(postgres_dsn, setup=True) as runtime:
        monkeypatch.setattr(main, "compiled_graph", runtime.graph)
        monkeypatch.setattr(main, "thread_retention", runtime.retention)
        first = main.ask(
            AskRequest(question="Original question", thread_id=thread_id),
            principal_id="individual:alice",
        )
        assert first.answer == "first"
        with pytest.raises(HTTPException) as wrong_repo:
            main.ask(
                AskRequest(question="Wrong repo", thread_id=thread_id, repo_id="other"),
                principal_id="individual:alice",
            )
        assert wrong_repo.value.status_code == 409
        with pytest.raises(HTTPException) as wrong_owner:
            main.ask(
                AskRequest(question="Wrong owner", thread_id=thread_id),
                principal_id="individual:bob",
            )
        assert wrong_owner.value.status_code == 404
        assert llm.invocations == 1

        # Age only this test thread in the disposable integration database.
        with psycopg.connect(postgres_dsn, autocommit=True) as conn:
            conn.execute(
                "UPDATE overture_conversations SET last_used = NOW() - "
                "(%s * INTERVAL '1 second') WHERE thread_id = %s",
                (main.settings.thread_ttl_seconds + 1, thread_id),
            )
        renewed = main.ask(
            AskRequest(question="New question", thread_id=thread_id, repo_id="other"),
            principal_id="individual:bob",
        )
        assert renewed.answer == "new"
        contents = [
            message.content
            for message in runtime.graph.get_state(
                {"configurable": {"thread_id": thread_id}}
            ).values["messages"]
        ]
        assert "Original question" not in contents
        assert "New question" in contents


def test_durable_lru_deletes_checkpoint_and_owner(postgres_dsn, monkeypatch):
    from app.persistence import open_postgres_runtime

    first_id = f"lru-first-{uuid.uuid4().hex}"
    second_id = f"lru-second-{uuid.uuid4().hex}"
    monkeypatch.setattr(main.settings, "checkpointer_backend", "postgres")
    monkeypatch.setattr(main.settings, "max_threads", 1)
    llm = FakeReActLLM(
        [AIMessage(content="first"), AIMessage(content="second"), AIMessage(content="new")]
    )
    monkeypatch.setattr("app.graph.get_llm", lambda: llm)
    with open_postgres_runtime(postgres_dsn, setup=True) as runtime:
        monkeypatch.setattr(main, "compiled_graph", runtime.graph)
        monkeypatch.setattr(main, "thread_retention", runtime.retention)
        for thread_id in (first_id, second_id):
            main.ask(
                AskRequest(question="Initial question", thread_id=thread_id),
                principal_id="individual:alice",
            )
        assert not runtime.graph.get_state(
            {"configurable": {"thread_id": first_id}}
        ).values
        replacement = main.ask(
            AskRequest(question="Replacement", thread_id=first_id),
            principal_id="individual:bob",
        )
        assert replacement.answer == "new"


def test_lifespan_selects_and_closes_postgres_runtime(monkeypatch):
    monkeypatch.setattr(main.settings, "checkpointer_backend", "postgres")
    monkeypatch.setattr(main.settings, "postgres_dsn", "postgresql://example")
    monkeypatch.setattr(main.settings, "postgres_setup", False)
    monkeypatch.setattr(main, "ensure_repo", lambda *args: None)
    monkeypatch.setattr(main, "load_portfolio_repos", lambda *args: [])
    original = (main.checkpointer, main.compiled_graph, main.thread_retention)
    runtime = SimpleNamespace(checkpointer=object(), graph=object(), retention=object())
    events = []

    @contextmanager
    def fake_open(dsn, *, setup):
        assert dsn == "postgresql://example"
        assert setup is False
        events.append("open")
        yield runtime
        events.append("close")

    monkeypatch.setattr("app.persistence.open_postgres_runtime", fake_open)

    async def exercise():
        async with main.lifespan(main.app):
            assert (main.checkpointer, main.compiled_graph, main.thread_retention) == (
                runtime.checkpointer,
                runtime.graph,
                runtime.retention,
            )

    asyncio.run(exercise())
    assert events == ["open", "close"]
    assert (main.checkpointer, main.compiled_graph, main.thread_retention) == original


def test_storage_failure_rejects_before_checkpoint_read(monkeypatch):
    class FailingRetention:
        def begin(self, *args):
            raise StorageUnavailable()

    monkeypatch.setattr(main, "thread_retention", FailingRetention())
    with (
        patch.object(main.compiled_graph, "get_state") as get_state,
        pytest.raises(HTTPException) as failed,
    ):
        main.ask(AskRequest(question="Hello there"), principal_id="individual:alice")
    assert failed.value.status_code == 503
    assert failed.value.detail == "Conversation storage unavailable"
    get_state.assert_not_called()


def test_settings_repr_omits_credentials():
    configured = Settings(
        _env_file=None,
        llm_api_key="model-private-value",
        api_key="shared-private-value",
        principal_api_keys={"alice": "user-private-value"},
        postgres_dsn="postgresql://private-value@localhost/db",
    )
    representation = repr(configured)
    assert "private-value" not in representation
