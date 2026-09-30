from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = "changeme"
    max_iterations: int = 5
    repo_path: str = "/data/repo"
    repo_git_url: str = ""
    log_level: str = "INFO"
    log_content_max_chars: int = Field(default=200, gt=0, le=1000)
    # Explicit diagnostic opt-in may include question text, tool arguments and
    # exception text in logs. The default only emits operational metadata.
    log_diagnostics_enabled: bool = False
    api_key: str = ""
    max_history_messages: int = 20
    # Conversations idle longer than this, or beyond max_threads (least recently
    # used first), are deleted from memory.
    thread_ttl_seconds: int = Field(default=24 * 60 * 60, gt=0)
    max_threads: int = Field(default=500, gt=0)
    portfolio_repos_path: str = "portfolio_repos.yaml"
    repo_root: str = "/data/repos"
    semantic_search_enabled: bool = False
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None


settings = Settings()
