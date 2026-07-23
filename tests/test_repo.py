import subprocess
from unittest.mock import patch

import pytest

from app.repo import ensure_repo


@pytest.fixture
def no_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr("app.repo.subprocess.run", forbidden)


class TestEnsureRepo:
    def test_existing_non_empty_repo_is_used_without_cloning(
        self, tmp_path, no_subprocess
    ):
        (tmp_path / "main.py").write_text("print('hi')\n")

        ensure_repo(str(tmp_path), "https://example.com/repo.git")

    def test_missing_repo_without_url_does_not_clone_or_raise(
        self, tmp_path, no_subprocess
    ):
        ensure_repo(str(tmp_path / "repo"), "")

    def test_empty_dir_with_url_triggers_shallow_clone(self, tmp_path):
        repo_path = tmp_path / "data" / "repo"
        git_url = "https://example.com/repo.git"

        with patch("app.repo.subprocess.run") as fake_run:
            ensure_repo(str(repo_path), git_url)

        fake_run.assert_called_once()
        command = fake_run.call_args.args[0]
        assert command == ["git", "clone", "--depth", "1", git_url, str(repo_path)]
        assert fake_run.call_args.kwargs["check"] is True
        assert repo_path.parent.is_dir()

    def test_clone_failure_aborts_startup(self, tmp_path):
        repo_path = tmp_path / "repo"
        error = subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "clone"],
            stderr="fatal: repository not found",
        )

        with (
            patch("app.repo.subprocess.run", side_effect=error),
            pytest.raises(RuntimeError, match="Failed to clone target repository"),
        ):
            ensure_repo(str(repo_path), "https://example.com/missing.git")

    def test_clone_timeout_aborts_startup(self, tmp_path):
        error = subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=120)

        with (
            patch("app.repo.subprocess.run", side_effect=error),
            pytest.raises(RuntimeError, match="Failed to clone target repository"),
        ):
            ensure_repo(str(tmp_path / "repo"), "https://example.com/slow.git")


class TestLifespanIntegration:
    def test_app_startup_calls_ensure_repo(self):
        from fastapi.testclient import TestClient

        from app.main import app

        with patch("app.main.ensure_repo") as fake_ensure, patch(
            "app.repo.ensure_repo"
        ):
            with TestClient(app):
                pass

        fake_ensure.assert_called_once()
