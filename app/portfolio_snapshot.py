"""Materialize and record the curated repository snapshots during image build."""

import argparse
import json
from pathlib import Path

from app.portfolio import load_portfolio_repos
from app.repo import ensure_repo


def build_portfolio_snapshot(catalog: Path, repo_root: Path, manifest: Path) -> dict:
    repos = load_portfolio_repos(str(catalog))
    if not repos:
        raise ValueError("Portfolio catalog is missing or empty")

    snapshots = []
    for repo in repos:
        if repo.revision is None:
            raise ValueError(f"Portfolio repository {repo.repo_id!r} has no pinned revision")
        repo_path = repo_root / repo.repo_id
        ensure_repo(str(repo_path), repo.git_url, revision=repo.revision)
        snapshots.append({"repo_id": repo.repo_id, "revision": repo.revision})

    result = {"repos": snapshots}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("portfolio_repos.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("/data/repos"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("portfolio_manifest.json")
    )
    args = parser.parse_args()
    build_portfolio_snapshot(args.catalog, args.repo_root, args.manifest)


if __name__ == "__main__":
    main()
