"""Deterministic admission and provider-budget regression tests for issue #41."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app import agent_tools
from app import graph as graph_module
from app.config import Settings, settings
from app.main import SUMMARIZATION_INSTRUCTION, _summarize_fn, app
from app.usage import (
    AdmissionController,
    ModelInputTooLarge,
    QuotaExceeded,
    RequestDeadlineExceeded,
    request_deadline,
)
from tests.conftest import TEST_API_KEY
from tests.test_health import FakeReActLLM


def test_burst_limits_per_client_and_globally():
    limiter = AdmissionController(
        per_client_rate=2,
        global_rate=3,
        window_seconds=10,
        per_client_concurrency=2,
        global_concurrency=3,
    )
    for client in ("a", "a", "b"):
        limiter.acquire(client, now=100).close()
    with pytest.raises(QuotaExceeded) as error:
        limiter.acquire("a", now=101)
    assert error.value.retry_after == 9
    with pytest.raises(QuotaExceeded):
        limiter.acquire("c", now=101)
    limiter.acquire("a", now=110).close()


def test_simultaneous_calls_respect_capacity_and_release_after_error():
    limiter = AdmissionController(
        per_client_rate=10,
        global_rate=10,
        window_seconds=60,
        per_client_concurrency=1,
        global_concurrency=2,
    )
    ready = threading.Event()
    release = threading.Event()

    def hold():
        with limiter.acquire("a", now=100):
            ready.set()
            release.wait(timeout=3)
            raise RuntimeError("provider failed")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(hold)
        assert ready.wait(timeout=3)
        with pytest.raises(QuotaExceeded):
            limiter.acquire("a", now=101)
        with limiter.acquire("b", now=101), pytest.raises(QuotaExceeded):
            limiter.acquire("c", now=101)
        release.set()
        with pytest.raises(RuntimeError):
            future.result(timeout=3)
    limiter.acquire("a", now=102).close()


def test_rejection_at_http_boundary_never_enters_graph(client, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    limiter = AdmissionController(
        per_client_rate=1,
        global_rate=1,
        window_seconds=60,
        per_client_concurrency=1,
        global_concurrency=1,
    )
    monkeypatch.setattr(main, "admission_controller", limiter)
    monkeypatch.setattr(settings, "api_key", TEST_API_KEY)
    fake_state = {
        "final_answer": "ok",
        "outcome": None,
        "trajectory": [],
        "iterations": 0,
    }
    with patch("app.main.compiled_graph") as graph:
        graph.invoke.return_value = fake_state
        first = client.post("/ask", json={"question": "What files exist?"})
        second = TestClient(app, headers={"X-API-Key": TEST_API_KEY}).post(
            "/ask", json={"question": "What files exist?"}
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"
    graph.invoke.assert_called_once()


@pytest.mark.parametrize(
    "field,value",
    [
        ("ask_rate_per_client", 0),
        ("ask_rate_global", 0),
        ("ask_concurrency_per_client", 0),
        ("ask_concurrency_global", 0),
        ("ask_rate_window_seconds", 0),
        ("provider_timeout_seconds", 0),
        ("provider_max_retries", -1),
        ("model_max_completion_tokens", 0),
        ("model_max_input_chars", 0),
        ("ask_deadline_seconds", 0),
        ("max_iterations", 0),
        ("max_history_messages", 0),
    ],
)
def test_invalid_limits_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_chat_and_embeddings_share_explicit_timeout_and_retry_policy(monkeypatch):
    chat_args = {}
    embedding_args = {}

    class FakeChat:
        def __init__(self, **kwargs):
            chat_args.update(kwargs)

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            embedding_args.update(kwargs)

        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(graph_module, "ChatOpenAI", FakeChat)
    monkeypatch.setattr(agent_tools, "OpenAIEmbeddings", FakeEmbeddings)
    token = request_deadline.set(time.monotonic() + 5)
    try:
        graph_module.get_llm()
        agent_tools._embed_fn(["hello"])
    finally:
        request_deadline.reset(token)

    assert 0 < chat_args["timeout"] <= 5
    assert 0 < embedding_args["timeout"] <= 5
    assert chat_args["max_retries"] == embedding_args["max_retries"] == 0
    assert chat_args["max_completion_tokens"] == settings.model_max_completion_tokens


def test_expired_deadline_prevents_chat_embedding_and_summary_calls(monkeypatch):
    chat = []
    embedding = []
    monkeypatch.setattr(
        graph_module, "ChatOpenAI", lambda **kwargs: chat.append(kwargs)
    )
    monkeypatch.setattr(
        agent_tools, "OpenAIEmbeddings", lambda **kwargs: embedding.append(kwargs)
    )
    token = request_deadline.set(time.monotonic() - 1)
    try:
        with pytest.raises(RequestDeadlineExceeded):
            graph_module.get_llm()
        with pytest.raises(RequestDeadlineExceeded):
            agent_tools._embed_fn(["hello"])
        with pytest.raises(RequestDeadlineExceeded):
            _summarize_fn("previous conversation")
    finally:
        request_deadline.reset(token)
    assert not chat and not embedding


def test_oversized_summary_and_embedding_input_never_reach_provider(monkeypatch):
    monkeypatch.setattr(
        settings, "model_max_input_chars", len(SUMMARIZATION_INSTRUCTION) + 10
    )
    with patch("app.graph.get_llm") as chat:
        with pytest.raises(ModelInputTooLarge):
            _summarize_fn("x" * 100)
        chat.assert_not_called()
    monkeypatch.setattr(settings, "model_max_input_chars", 5)
    with patch("app.agent_tools.OpenAIEmbeddings") as embedding:
        with pytest.raises(ModelInputTooLarge):
            agent_tools._embed_fn(["long input"])
        embedding.assert_not_called()


def test_oversized_chat_context_returns_413_without_model_invocation(
    client, monkeypatch
):
    fake_llm = FakeReActLLM([AIMessage(content="should not run")])
    monkeypatch.setattr(settings, "model_max_input_chars", 10)
    monkeypatch.setattr(graph_module, "get_llm", lambda: fake_llm)

    response = client.post("/ask", json={"question": "Explain this repository"})

    assert response.status_code == 413
    assert fake_llm.invocations == 0


def test_expired_deadline_returns_504_and_releases_capacity(client, monkeypatch):
    from app import main

    limiter = AdmissionController(
        per_client_rate=2,
        global_rate=2,
        window_seconds=60,
        per_client_concurrency=1,
        global_concurrency=1,
    )
    monkeypatch.setattr(main, "admission_controller", limiter)
    monkeypatch.setattr(settings, "ask_deadline_seconds", 0.000001)
    with patch("app.graph.get_llm", side_effect=RequestDeadlineExceeded):
        response = client.post("/ask", json={"question": "Explain this repository"})

    assert response.status_code == 504
    assert limiter.active_global == 0
