import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


SHARED_PRINCIPAL = "shared-study-key"


def require_api_key(provided: str | None = Security(api_key_header)) -> str:
    """Authenticate the caller and return a server-verified principal ID."""
    if settings.auth_mode == "individual":
        keys = settings.principal_api_keys
        if not keys or any(not name or not key for name, key in keys.items()) or len(
            set(keys.values())
        ) != len(keys):
            raise HTTPException(status_code=503, detail="Authentication is not configured")
        # Compare all keys without exposing which principal matched.
        matched = [
            name for name, key in keys.items()
            if provided is not None and secrets.compare_digest(provided, key)
        ]
        if not matched:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return f"individual:{matched[0]}"
    if not settings.api_key:
        raise HTTPException(
            status_code=503,
            detail="API key authentication is not configured on the server",
        )
    if not provided or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return SHARED_PRINCIPAL
