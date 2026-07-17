import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

from app.config import settings

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def clip(text: str) -> str:
    """Trunca conteúdo vindo do usuário/LLM antes de logar."""
    limit = settings.log_content_max_chars
    if len(text) <= limit:
        return text
    return text[:limit] + "... [truncated]"

_LOG_RECORD_DEFAULT_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None))
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line.

    Fields passed via ``logger.info(..., extra={...})`` are merged into the
    payload, so call sites can attach structured data without string
    formatting.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_DEFAULT_ATTRS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Send all ``app.*`` logs to stdout as JSON lines."""
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level.upper())
    app_logger.propagate = False

    if any(handler.name == "overture-json" for handler in app_logger.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.name = "overture-json"
    handler.setFormatter(JsonFormatter())
    app_logger.addHandler(handler)
