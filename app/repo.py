import logging
import subprocess
from pathlib import Path

from app.portfolio import PortfolioRepo

logger = logging.getLogger(__name__)

CLONE_TIMEOUT_SECONDS = 120


def ensure_repo(repo_path: str, git_url: str) -> None:
    """Make sure the target repository exists at repo_path.

    An existing non-empty directory is used as-is (local dev with a manually
    provisioned repo). Otherwise, when git_url is set, the repository is
    shallow-cloned into repo_path. A configured URL that fails to clone aborts
    startup: serving /ask without a repository would fail every tool call.
    """
    path = Path(repo_path)
    if path.is_dir() and any(path.iterdir()):
        logger.info(
            "repo_ready", extra={"repo_path": repo_path, "source": "existing"}
        )
        return

    if not git_url:
        logger.warning(
            "repo_missing",
            extra={
                "repo_path": repo_path,
                "hint": "set APP_REPO_GIT_URL or provision the path manually",
            },
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", git_url, repo_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        stderr = (getattr(exc, "stderr", "") or "").strip()
        logger.error(
            "repo_clone_failed",
            extra={
                "repo_path": repo_path,
                "git_url": git_url,
                "error": stderr or str(exc),
            },
        )
        raise RuntimeError(
            f"Failed to clone target repository from {git_url}"
        ) from exc

    logger.info(
        "repo_cloned",
        extra={"repo_path": repo_path, "git_url": git_url, "source": "clone"},
    )


def build_repo_registry(
    repos: list[PortfolioRepo], repo_root: str
) -> dict[str, str]:
    """Materialize each curated portfolio repo on disk and build a registry.

    A repo that fails to clone is logged and excluded from the registry
    rather than aborting startup: one broken portfolio entry shouldn't take
    down the whole app, unlike the single required default repo.
    """
    registry: dict[str, str] = {}
    for repo in repos:
        repo_path = str(Path(repo_root) / repo.repo_id)
        try:
            ensure_repo(repo_path, repo.git_url)
        except RuntimeError:
            logger.error(
                "portfolio_repo_skipped",
                extra={"repo_id": repo.repo_id, "git_url": repo.git_url},
            )
            continue

        path = Path(repo_path)
        if path.is_dir() and any(path.iterdir()):
            registry[repo.repo_id] = repo_path
        else:
            logger.error(
                "portfolio_repo_skipped",
                extra={"repo_id": repo.repo_id, "git_url": repo.git_url},
            )

    return registry
