import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

from app.config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def clip(text: str) -> str:
    """Bound content explicitly enabled for diagnostic logging."""
    limit = settings.log_content_max_chars
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"

# Each event declares the operational fields it may expose. Adding a new log
# field requires an explicit decision here rather than relying on arbitrary
# ``extra`` values (which may contain questions, URLs or provider exceptions).
_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "repo_ready": frozenset({"source"}),
    "repo_missing": frozenset(),
    "repo_cloned": frozenset({"source", "revision"}),
    "repo_clone_failed": frozenset(
        {"status", "error_type", "returncode", "duration_ms"}
    ),
    "portfolio_repo_skipped": frozenset({"repo_id", "error_type"}),
    "route_selected": frozenset({"route", "requested_tools", "iterations"}),
    "tool_executed": frozenset(
        {"tool", "status", "error_type", "error_code", "duration_ms"}
    ),
    "budget_exceeded": frozenset(
        {"requested_tool_calls", "remaining_budget", "max_iterations"}
    ),
    "ask_completed": frozenset(
        {"tools_called", "iterations", "outcome", "duration_ms"}
    ),
    "ask_failed": frozenset({"status", "error_type", "duration_ms"}),
    "summarization_failed": frozenset({"status", "error_type", "duration_ms"}),
    "semantic_index_built": frozenset(
        {"indexed_files", "partial_files", "skipped_files"}
    ),
    "semantic_search_unavailable": frozenset(
        {"status", "error_type", "duration_ms"}
    ),
}
_DIAGNOSTIC_FIELDS = frozenset({"question", "tool_input", "output_summary", "error"})
_TOOL_NAMES = frozenset(
    {"list_files", "read_file", "grep_repo", "semantic_search", "agent_decide", "max_iterations_guardrail"}
)
_MAX_LIST_ITEMS = 16


def safe_tool_name(name: str) -> str:
    """Avoid copying arbitrary provider-supplied tool names into telemetry."""
    return name if name in _TOOL_NAMES else "unknown_tool"


def _bounded_field(key: str, value):
    limit = settings.log_content_max_chars
    if key in {"tool", "requested_tools", "tools_called"}:
        if isinstance(value, list):
            return [safe_tool_name(str(item)) for item in value[:_MAX_LIST_ITEMS]]
        return safe_tool_name(str(value))
    if isinstance(value, str):
        return value[:limit] + ("... [truncated]" if len(value) > limit else "")
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_bounded_field(key, item) for item in value[:_MAX_LIST_ITEMS]]
    return str(value)[:limit]


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line.

    Only allowlisted operational fields and explicitly enabled diagnostic
    fields are emitted from ``logger.info(..., extra={...})``.
    """

    def format(self, record: logging.LogRecord) -> str:
        event = record.getMessage()
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": event if event in _EVENT_FIELDS else "unclassified_event",
        }

        # Capture at LogRecord creation too: queued/deferred handlers may format
        # after the request's ContextVar has been reset.
        request_id = getattr(record, "_overture_request_id", None) or request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id[:64]

        fields = _EVENT_FIELDS.get(event, frozenset())
        if settings.log_diagnostics_enabled:
            fields = fields | _DIAGNOSTIC_FIELDS
        for key in fields:
            if key in record.__dict__:
                payload[key] = _bounded_field(key, record.__dict__[key])

        if settings.log_diagnostics_enabled and record.exc_info:
            stack = self.formatException(record.exc_info)
            payload["exception"] = stack[: settings.log_content_max_chars * 8]

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Send all ``app.*`` logs to stdout as JSON lines."""
    factory = logging.getLogRecordFactory()
    if not getattr(factory, "_overture_request_id_factory", False):
        def with_request_id(*args, **kwargs):
            record = factory(*args, **kwargs)
            record._overture_request_id = request_id_var.get()
            return record

        with_request_id._overture_request_id_factory = True
        logging.setLogRecordFactory(with_request_id)

    app_logger = logging.getLogger("app")
    app_logger.setLevel(level.upper())
    app_logger.propagate = False

    if any(handler.name == "overture-json" for handler in app_logger.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.name = "overture-json"
    handler.setFormatter(JsonFormatter())
    app_logger.addHandler(handler)
