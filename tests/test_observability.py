import json
import logging

import pytest
from langchain_core.messages import AIMessage

from app.config import settings
from app.observability import JsonFormatter, clip, configure_logging, request_id_var
from tests.test_health import FakeReActLLM


class CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured_app_logs():
    handler = CaptureHandler()
    app_logger = logging.getLogger("app")
    app_logger.addHandler(handler)
    yield handler.records
    app_logger.removeHandler(handler)


class TestJsonFormatter:
    def _format(self, record) -> dict:
        return json.loads(JsonFormatter().format(record))

    def test_includes_base_fields_and_extras(self):
        record = logging.LogRecord(
            name="app.graph",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="tool_executed",
            args=(),
            exc_info=None,
        )
        record.tool = "read_file"
        record.duration_ms = 12.3

        payload = self._format(record)

        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.graph"
        assert payload["event"] == "tool_executed"
        assert payload["tool"] == "read_file"
        assert payload["duration_ms"] == 12.3
        assert "timestamp" in payload
        assert "request_id" not in payload

    def test_includes_request_id_when_set(self):
        record = logging.LogRecord(
            name="app.main",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="ask_completed",
            args=(),
            exc_info=None,
        )

        token = request_id_var.set("req-123")
        try:
            payload = self._format(record)
        finally:
            request_id_var.reset(token)

        assert payload["request_id"] == "req-123"


class TestConfigureLogging:
    def test_is_idempotent(self):
        configure_logging()
        configure_logging()

        app_logger = logging.getLogger("app")
        json_handlers = [
            handler
            for handler in app_logger.handlers
            if handler.name == "overture-json"
        ]
        assert len(json_handlers) == 1


class TestClip:
    def test_short_text_is_unchanged(self):
        assert clip("short question") == "short question"

    def test_long_text_is_truncated_with_marker(self, monkeypatch):
        monkeypatch.setattr(settings, "log_content_max_chars", 10)

        result = clip("x" * 50)

        assert result == "x" * 10 + "... [truncated]"

    def test_logged_question_is_truncated(
        self, client, monkeypatch, captured_app_logs
    ):
        monkeypatch.setattr(settings, "log_content_max_chars", 20)

        def broken_llm():
            raise RuntimeError("llm exploded")

        monkeypatch.setattr("app.graph.get_llm", broken_llm)

        long_question = "why does " + "q" * 100 + " happen?"
        client.post("/ask", json={"question": long_question})

        events = {record.getMessage(): record for record in captured_app_logs}
        logged = events["ask_failed"].question
        assert logged == long_question[:20] + "... [truncated]"


class TestAskRequestLogging:
    def test_successful_request_logs_completion_event(
        self, client, fake_repo, monkeypatch, captured_app_logs
    ):
        fake_llm = FakeReActLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"relative_path": "src/main.py"},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="answer after reading"),
            ]
        )
        monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)
        monkeypatch.setattr("app.main.settings.repo_path", str(fake_repo))

        response = client.post("/ask", json={"question": "Read src/main.py"})
        assert response.status_code == 200

        events = {record.getMessage(): record for record in captured_app_logs}

        completed = events["ask_completed"]
        assert completed.question == "Read src/main.py"
        assert completed.tools_called == ["read_file", "agent_decide"]
        assert completed.iterations == 1
        assert completed.outcome == "answered"
        assert completed.duration_ms >= 0

        tool_record = events["tool_executed"]
        assert tool_record.tool == "read_file"
        assert tool_record.status == "ok"
        assert tool_record.duration_ms >= 0

        assert "route_selected" in events

    def test_failed_request_logs_error_event(
        self, client, monkeypatch, captured_app_logs
    ):
        def broken_llm():
            raise RuntimeError("llm exploded")

        monkeypatch.setattr("app.graph.get_llm", broken_llm)

        response = client.post("/ask", json={"question": "Anything"})
        assert response.status_code == 500

        events = {record.getMessage(): record for record in captured_app_logs}
        failed = events["ask_failed"]
        assert failed.levelno == logging.ERROR
        assert failed.question == "Anything"
        assert "llm exploded" in failed.error
        assert failed.duration_ms >= 0
        assert "ask_completed" not in events
