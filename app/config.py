from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = Field(default="changeme", repr=False)
    max_iterations: int = Field(default=5, gt=0)
    ask_rate_per_client: int = Field(default=30, gt=0)
    ask_rate_global: int = Field(default=120, gt=0)
    ask_rate_window_seconds: int = Field(default=60, gt=0)
    ask_concurrency_per_client: int = Field(default=2, gt=0)
    ask_concurrency_global: int = Field(default=8, gt=0)
    ask_deadline_seconds: float = Field(default=60, gt=0)
    provider_timeout_seconds: float = Field(default=20, gt=0)
    provider_max_retries: int = Field(default=0, ge=0, le=5)
    model_max_completion_tokens: int = Field(default=1024, gt=0)
    model_max_input_chars: int = Field(default=100_000, gt=0)
    repo_path: str = "/data/repo"
    repo_git_url: str = ""
    log_level: str = "INFO"
    log_content_max_chars: int = Field(default=200, gt=0, le=1000)
    # Explicit diagnostic opt-in may include question text, tool arguments and
    # exception text in logs. The default only emits operational metadata.
    log_diagnostics_enabled: bool = False
    api_key: str = Field(default="", repr=False)
    # shared preserves the single-key study deployment; individual keys identify
    # distinct callers and must be kept by a trusted server, never browser code.
    auth_mode: Literal["shared", "individual"] = "shared"
    principal_api_keys: dict[str, str] = Field(default_factory=dict, repr=False)
    max_history_messages: int = Field(default=20, gt=0)
    # Conversations idle longer than this, or beyond max_threads (least recently
    # used first), are deleted from memory.
    thread_ttl_seconds: int = Field(default=24 * 60 * 60, gt=0)
    max_threads: int = Field(default=500, gt=0)
    checkpointer_backend: Literal["memory", "postgres"] = "memory"
    postgres_dsn: str = Field(default="", repr=False)
    # Apply LangGraph's and Overture's additive schema setup only when opted in.
    postgres_setup: bool = False
    postgres_pool_max_size: int = Field(default=20, ge=4)
    portfolio_repos_path: str = "portfolio_repos.yaml"
    repo_root: str = "/data/repos"
    semantic_search_enabled: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = Field(default=None, repr=False)


settings = Settings()
