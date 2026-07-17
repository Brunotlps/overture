import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(api_key_header)) -> None:
    """Valida a API key antes de qualquer trabalho do agente (fail-closed)."""
    if not settings.api_key:
        raise HTTPException(
            status_code=503,
            detail="API key authentication is not configured on the server",
        )
    if not provided or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
