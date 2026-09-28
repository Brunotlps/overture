"""Offline revision and manifest checks using a tiny local Git repository."""

import json
import subprocess

import pytest
import yaml

from app.portfolio import load_portfolio_repos
from app.portfolio_snapshot import build_portfolio_snapshot
from app.repo import ensure_repo


def _git(*args, cwd=None):
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def local_origin(tmp_path):
    origin = tmp_path / "origin"
    _git("init", "-q", str(origin))
    _git("config", "user.name", "Test", cwd=origin)
    _git("config", "user.email", "test@example.com", cwd=origin)
    file = origin / "README.md"
    file.write_text("first snapshot\n")
    _git("add", "README.md", cwd=origin)
    _git("commit", "-qm", "first", cwd=origin)
    first = _git("rev-parse", "HEAD", cwd=origin)
    file.write_text("second snapshot\n")
    _git("commit", "-qam", "second", cwd=origin)
    second = _git("rev-parse", "HEAD", cwd=origin)
    return origin, first, second


def _catalog(path, origin, revision):
    path.write_text(
        yaml.safe_dump(
            {
                "repos": [
                    {
                        "repo_id": "sample",
                        "git_url": str(origin),
                        "display_name": "Sample",
                        "revision": revision,
                    }
                ]
            }
        )
    )


def test_pinned_snapshots_are_repeatable_and_recorded(tmp_path, local_origin):
    origin, first, second = local_origin
    catalog = tmp_path / "catalog.yaml"
    manifest = tmp_path / "manifest.json"
    first_root = tmp_path / "first-build"
    _catalog(catalog, origin, first)

    result = build_portfolio_snapshot(catalog, first_root, manifest)
    build_portfolio_snapshot(catalog, first_root, manifest)

    assert result == {"repos": [{"repo_id": "sample", "revision": first}]}
    assert json.loads(manifest.read_text()) == result
    assert (first_root / "sample" / "README.md").read_text() == "first snapshot\n"
    assert _git("rev-parse", "HEAD", cwd=first_root / "sample") == first

    _catalog(catalog, origin, second)
    second_root = tmp_path / "second-build"
    build_portfolio_snapshot(catalog, second_root, tmp_path / "second-manifest.json")
    assert (second_root / "sample" / "README.md").read_text() == "second snapshot\n"

    with pytest.raises(RuntimeError, match="revision or origin"):
        build_portfolio_snapshot(catalog, first_root, manifest)


def test_invalid_revision_fails_before_manifest_or_partial_checkout(tmp_path, local_origin):
    origin, _, _ = local_origin
    catalog = tmp_path / "catalog.yaml"
    root = tmp_path / "repos"
    manifest = tmp_path / "manifest.json"
    _catalog(catalog, origin, "0" * 40)

    with pytest.raises(RuntimeError, match="materialize"):
        build_portfolio_snapshot(catalog, root, manifest)

    assert not manifest.exists()
    assert not (root / "sample").exists()


def test_dirty_or_wrong_origin_is_never_reused(tmp_path, local_origin):
    origin, first, _ = local_origin
    destination = tmp_path / "repos" / "sample"
    ensure_repo(str(destination), str(origin), revision=first)
    (destination / "README.md").write_text("changed after build\n")

    with pytest.raises(RuntimeError, match="local changes"):
        ensure_repo(str(destination), str(origin), revision=first)

    _git("checkout", "--", "README.md", cwd=destination)
    with pytest.raises(RuntimeError, match="revision or origin"):
        ensure_repo(str(destination), str(tmp_path / "wrong-origin"), revision=first)


def test_catalog_rejects_short_revision_and_image_requires_a_pin(tmp_path, local_origin):
    origin, _, _ = local_origin
    catalog = tmp_path / "catalog.yaml"
    _catalog(catalog, origin, "short")
    with pytest.raises(ValueError, match="Invalid revision"):
        load_portfolio_repos(str(catalog))

    _catalog(catalog, origin, 123)
    with pytest.raises(ValueError, match="Invalid revision"):
        load_portfolio_repos(str(catalog))

    catalog.write_text(
        yaml.safe_dump(
            {
                "repos": [
                    {
                        "repo_id": "sample",
                        "git_url": str(origin),
                        "display_name": "Sample",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="no pinned revision"):
        build_portfolio_snapshot(catalog, tmp_path / "repos", tmp_path / "manifest")
