import logging
import subprocess
import tempfile
from pathlib import Path

from app.portfolio import PortfolioRepo

logger = logging.getLogger(__name__)

CLONE_TIMEOUT_SECONDS = 120


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        raise RuntimeError("Git could not materialize the configured revision") from None
    return result.stdout.strip()


def _verify_pinned_repo(path: Path, git_url: str, revision: str) -> None:
    if not (path / ".git").exists():
        raise RuntimeError(f"Pinned repository at {path} is not a Git checkout")
    actual_revision = _git("-C", str(path), "rev-parse", "HEAD")
    actual_origin = _git("-C", str(path), "config", "--get", "remote.origin.url")
    if actual_revision != revision or actual_origin != git_url:
        raise RuntimeError(f"Pinned repository at {path} does not match its revision or origin")
    if _git("-C", str(path), "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError(f"Pinned repository at {path} contains local changes")


def _ensure_pinned_repo(path: Path, git_url: str, revision: str) -> None:
    if not git_url:
        raise RuntimeError("Pinned repository requires a Git URL")
    if path.is_dir() and any(path.iterdir()):
        _verify_pinned_repo(path, git_url, revision)
        logger.info("repo_ready", extra={"repo_path": str(path), "source": "pinned"})
        return
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Repository path is not a directory: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portfolio-clone-", dir=path.parent) as temp:
        clone_path = Path(temp) / "repo"
        try:
            _git("clone", "--filter=blob:none", "--no-checkout", git_url, str(clone_path))
            _git("-C", str(clone_path), "checkout", "--detach", revision)
            _verify_pinned_repo(clone_path, git_url, revision)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not materialize revision {revision} for {path.name}: {exc}"
            ) from None
        if path.exists():
            path.rmdir()
        clone_path.rename(path)
    logger.info(
        "repo_cloned",
        extra={"repo_path": str(path), "source": "pinned", "revision": revision},
    )


def ensure_repo(repo_path: str, git_url: str, revision: str | None = None) -> None:
    """Make sure the target repository exists at repo_path.

    A pinned repository must match both the configured commit and origin.
    Unpinned repositories retain the local-development and shallow-clone path.
    """
    path = Path(repo_path)
    if revision is not None:
        _ensure_pinned_repo(path, git_url, revision)
        return

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
            ensure_repo(repo_path, repo.git_url, revision=repo.revision)
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
