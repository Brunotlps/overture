"""Regression checks for error and telemetry privacy using synthetic content."""

import json
import logging
import subprocess
import sys
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, ToolMessage
from openai import OpenAIError
from pydantic import ValidationError

from app.config import Settings, settings
from app.graph import execute_tools_node, generate_response_node
from app.main import ask, compiled_graph
from app.observability import JsonFormatter, request_id_var
from app.repo import ensure_repo
from app.schemas import AskRequest, Category
from tests.test_health import FakeReActLLM

MARKER = "SYNTHETIC_PRIVATE_MARKER_46"


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def records():
    handler = Capture()
    app_logger = logging.getLogger("app")
    app_logger.addHandler(handler)
    yield handler.records
    app_logger.removeHandler(handler)


def _payloads(records, event):
    return [
        json.loads(JsonFormatter().format(record))
        for record in records
        if record.getMessage() == event
    ]


def _tool_call():
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"relative_path": "missing.py"}, "id": "call_46"}
        ],
    )


class BrokenTool:
    def invoke(self, args):
        raise ValueError(f"provider detail {MARKER}")


def test_private_log_mode_keeps_operational_fields_without_content(monkeypatch):
    monkeypatch.setattr(settings, "log_diagnostics_enabled", False)
    record = logging.LogRecord("app.main", logging.ERROR, "", 0, "ask_failed", (), None)
    record.question = MARKER
    record.tool_input = MARKER
    record.error = MARKER
    record.error_type = "RuntimeError"
    record.status = "error"
    record.duration_ms = 3.2
    record.extra_unknown = MARKER
    token = request_id_var.set("request-46")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert MARKER not in json.dumps(payload)
    assert payload["event"] == "ask_failed"
    assert payload["request_id"] == "request-46"
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
    assert payload["duration_ms"] == 3.2


def test_diagnostic_mode_is_explicit_bounded_and_never_logs_git_url(monkeypatch):
    monkeypatch.setattr(settings, "log_diagnostics_enabled", True)
    monkeypatch.setattr(settings, "log_content_max_chars", 12)
    record = logging.LogRecord("app.repo", logging.ERROR, "", 0, "repo_clone_failed", (), None)
    record.error_type = "CalledProcessError"
    record.git_url = f"https://user:{MARKER}@example.invalid/repo.git"
    record.error = MARKER
    try:
        raise RuntimeError(MARKER)
    except RuntimeError:
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))
    assert "git_url" not in payload
    assert payload["error"] == MARKER[:12] + "... [truncated]"
    assert len(payload["exception"]) <= 96


def test_unknown_event_and_provider_tool_names_are_bounded(monkeypatch):
    monkeypatch.setattr(settings, "log_diagnostics_enabled", False)
    record = logging.LogRecord("app.graph", logging.INFO, "", 0, MARKER, (), None)
    record.extra_unknown = MARKER
    assert json.loads(JsonFormatter().format(record))["event"] == "unclassified_event"

    record.msg = "route_selected"
    record.requested_tools = [MARKER] * 100
    payload = json.loads(JsonFormatter().format(record))
    assert payload["requested_tools"] == ["unknown_tool"] * 16
    assert MARKER not in json.dumps(payload)


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_invalid_log_content_limits_are_rejected(limit):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_content_max_chars=limit)


def test_handled_tool_error_is_safe_for_model_trajectory_and_log(records):
    state = {"messages": [_tool_call()], "repo_path": "/synthetic/repo"}
    with patch("app.graph.get_tool_registry", return_value={"read_file": BrokenTool()}):
        updates = execute_tools_node(state)

    message = updates["messages"][0]
    step = updates["trajectory"][0]
    event = _payloads(records, "tool_executed")[-1]
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_46"
    assert "invalid_input" in message.content
    assert MARKER not in str(message.content)
    assert MARKER not in step.model_dump_json()
    assert MARKER not in json.dumps(event)
    assert event["status"] == "error"
    assert event["error_type"] == "ValueError"
    assert event["duration_ms"] >= 0


def test_private_tool_log_omits_provider_arguments(records):
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"relative_path": f"{MARKER}.py"},
                "id": "private_args",
            }
        ],
    )
    with patch("app.graph.get_tool_registry", return_value={"read_file": BrokenTool()}):
        execute_tools_node({"messages": [call], "repo_path": "/synthetic/repo"})

    assert MARKER not in json.dumps(_payloads(records, "tool_executed"))
    assert "tool_input" not in records[-1].__dict__


def test_successful_ask_keeps_tool_protocol_without_leaking_error(monkeypatch):
    fake_llm = FakeReActLLM([_tool_call(), AIMessage(content="handled")])
    monkeypatch.setattr(settings, "repo_path", "/synthetic/repo")
    with (
        patch("app.graph.get_llm", return_value=fake_llm),
        patch("app.graph.get_tool_registry", return_value={"read_file": BrokenTool()}),
    ):
        response = ask(AskRequest(question="Explain behavior", thread_id="privacy-tool"))
        state = compiled_graph.get_state(
            {"configurable": {"thread_id": response.thread_id}}
        ).values

    assert response.answer == "handled"
    assert MARKER not in response.model_dump_json()
    assert any(isinstance(m, ToolMessage) for m in fake_llm.last_messages)
    assert not any(MARKER in str(m.content) for m in state["messages"])


def test_http_200_tool_error_has_safe_trajectory(client, monkeypatch, records):
    monkeypatch.setattr(settings, "repo_path", "/synthetic/repo")
    fake_llm = FakeReActLLM([_tool_call(), AIMessage(content="handled")])
    with (
        patch("app.graph.get_llm", return_value=fake_llm),
        patch("app.graph.get_tool_registry", return_value={"read_file": BrokenTool()}),
    ):
        response = client.post("/ask", json={"question": "Explain behavior"})

    assert response.status_code == 200
    assert MARKER not in response.text
    assert response.json()["trajectory"][0]["output_summary"] == "failed: invalid_input"
    assert MARKER not in json.dumps(_payloads(records, "tool_executed"))


def test_summary_failure_logs_type_and_request_id_without_exception(
    monkeypatch, records
):
    monkeypatch.setattr(settings, "repo_path", "/synthetic/repo")
    with patch("app.graph.get_llm", return_value=FakeReActLLM([AIMessage(content="ok")])):
        ask(AskRequest(question="First question", thread_id="privacy-summary"))
    records.clear()
    monkeypatch.setattr(settings, "max_history_messages", 1)
    with (
        patch("app.graph.get_llm", return_value=FakeReActLLM([AIMessage(content="ok")])),
        patch(
            "app.main.build_conversation_summary",
            side_effect=ValueError(f"summary failed {MARKER}"),
        ),
    ):
        response = ask(AskRequest(question="Follow up", thread_id="privacy-summary"))

    event = _payloads(records, "summarization_failed")[-1]
    assert response.answer == "ok"
    assert MARKER not in json.dumps(event)
    assert event["error_type"] == "ValueError"
    assert event["request_id"]


def test_unexpected_model_failure_keeps_generic_http_detail_and_private_log(
    monkeypatch, records
):
    monkeypatch.setattr(settings, "repo_path", "/synthetic/repo")
    with (
        patch("app.graph.get_llm", side_effect=RuntimeError(f"provider {MARKER}")),
        pytest.raises(HTTPException) as raised,
    ):
        ask(AskRequest(question=f"Investigate {MARKER}"))

    event = _payloads(records, "ask_failed")[-1]
    assert raised.value.status_code == 500
    assert MARKER not in str(raised.value.detail)
    assert MARKER not in json.dumps(event)
    assert event["error_type"] == "RuntimeError"
    assert event["duration_ms"] >= 0
    assert event["request_id"]


def test_http_500_hides_model_exception(client, monkeypatch, records):
    with patch("app.graph.get_llm", side_effect=RuntimeError(f"provider {MARKER}")):
        response = client.post("/ask", json={"question": f"Investigate {MARKER}"})

    assert response.status_code == 500
    assert MARKER not in response.text
    assert MARKER not in json.dumps(_payloads(records, "ask_failed"))


def test_clone_failure_does_not_log_or_raise_credential_url(tmp_path, records):
    url = f"https://user:{MARKER}@example.invalid/repo.git"
    failure = subprocess.CalledProcessError(
        128, ["git", "clone", url], stderr=f"fatal: {MARKER}"
    )
    with (
        patch("app.repo.subprocess.run", side_effect=failure),
        pytest.raises(RuntimeError) as raised,
    ):
        ensure_repo(str(tmp_path / "repo"), url)

    event = _payloads(records, "repo_clone_failed")[-1]
    assert MARKER not in str(raised.value)
    assert MARKER not in json.dumps(event)
    assert event["error_type"] == "CalledProcessError"
    assert event["status"] == "error"
    assert event["duration_ms"] >= 0


def test_legacy_graph_model_error_is_also_generic():
    class BrokenLLM:
        def invoke(self, prompt):
            raise OpenAIError(MARKER)

    state = {
        "user_input": "Explain behavior",
        "target": None,
        "category": Category.STRUCTURAL,
        "tool_output": "file contents",
    }
    with patch("app.graph.get_llm", return_value=BrokenLLM()):
        updates = generate_response_node(state)

    assert MARKER not in updates["final_answer"]
    assert MARKER not in updates["trajectory"][0].model_dump_json()
