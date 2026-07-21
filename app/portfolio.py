import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


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
        repos = [PortfolioRepo(**entry) for entry in data.get("repos", [])]
    except TypeError as exc:
        raise ValueError(f"Invalid portfolio repos entry in {path}: {exc}") from exc

    for repo in repos:
        if not REPO_ID_PATTERN.match(repo.repo_id):
            raise ValueError(
                f"Invalid repo_id {repo.repo_id!r} in {path}: must match "
                f"{REPO_ID_PATTERN.pattern}"
            )

    repo_ids = [repo.repo_id for repo in repos]
    if len(repo_ids) != len(set(repo_ids)):
        raise ValueError(f"Duplicate repo_id in {path}: {repo_ids}")

    return repos
