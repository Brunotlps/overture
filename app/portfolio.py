from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PortfolioRepo:
    repo_id: str
    git_url: str
    display_name: str


def load_portfolio_repos(path: str) -> list[PortfolioRepo]:
    """Load the curated portfolio repo list from a YAML config file.

    A missing file means the feature isn't configured: returns an empty
    list rather than failing, so deployments without it keep today's
    single-default-repo behavior.
    """
    if not Path(path).is_file():
        return []

    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed portfolio repos file at {path}: {exc}") from exc

    try:
        return [PortfolioRepo(**entry) for entry in data.get("repos", [])]
    except TypeError as exc:
        raise ValueError(f"Invalid portfolio repos entry in {path}: {exc}") from exc
