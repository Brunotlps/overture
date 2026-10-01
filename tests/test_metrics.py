"""Offline metric contract tests: bounded labels, auth, and request outcomes."""

from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import main
from app.main import app
from app.metrics import Metrics
from tests.test_health import FakeReActLLM


def test_metrics_endpoint_requires_key_and_exposes_only_app_metrics(
    client, monkeypatch
):
    monkeypatch.setattr(main, "metrics", Metrics())
    assert TestClient(app).get("/metrics").status_code == 401
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "overture_ask_requests_total" in response.text
    assert "python_gc_objects_collected" not in response.text


def test_answer_records_count_latency_and_iterations_without_content(
    client, monkeypatch
):
    metrics = Metrics()
    monkeypatch.setattr(main, "metrics", metrics)
    monkeypatch.setattr(
        "app.graph.get_llm", lambda: FakeReActLLM([AIMessage(content="answered")])
    )
    marker = "PRIVATE_QUESTION_MARKER"
    response = client.post(
        "/ask", json={"question": f"Explain {marker}", "thread_id": marker}
    )
    exported = client.get("/metrics").text

    assert response.status_code == 200
    assert 'overture_ask_requests_total{outcome="answered"} 1.0' in exported
    assert 'overture_ask_duration_seconds_count{outcome="answered"} 1.0' in exported
    assert 'overture_ask_iterations_count{outcome="answered"} 1.0' in exported
    assert marker not in exported


def test_graph_error_and_quota_rejection_are_counted_without_model_call(
    client, monkeypatch
):
    from app.usage import AdmissionController

    monkeypatch.setattr(main, "metrics", Metrics())
    monkeypatch.setattr(
        main,
        "admission_controller",
        AdmissionController(
            per_client_rate=1,
            global_rate=1,
            window_seconds=60,
            per_client_concurrency=1,
            global_concurrency=1,
        ),
    )
    with patch("app.main.compiled_graph") as graph:
        graph.invoke.side_effect = RuntimeError("provider down")
        failed = client.post("/ask", json={"question": "Explain code"})
        rejected = client.post("/ask", json={"question": "Explain code"})

    exported = client.get("/metrics").text
    assert failed.status_code == 500
    assert rejected.status_code == 429
    graph.invoke.assert_called_once()
    assert 'overture_ask_requests_total{outcome="error"} 1.0' in exported
    assert 'overture_ask_requests_total{outcome="rejected"} 1.0' in exported


def test_unknown_outcome_label_is_collapsed():
    metrics = Metrics()
    metrics.record_ask("untrusted-user-content", 0.2, 1)
    exported = metrics.render().decode()
    assert "untrusted-user-content" not in exported
    assert 'overture_ask_requests_total{outcome="other"} 1.0' in exported
