from app.portfolio import PortfolioRepo
from app.repo import build_repo_registry


def test_repo_that_clones_successfully_is_present_in_registry(tmp_path):
    repo_root = tmp_path / "repos"
    existing = repo_root / "overture"
    existing.mkdir(parents=True)
    (existing / "README.md").write_text("# Overture\n")

    repos = [
        PortfolioRepo(
            repo_id="overture",
            git_url="https://github.com/Brunotlps/overture",
            display_name="Overture",
        )
    ]

    registry = build_repo_registry(repos, str(repo_root))

    assert registry == {"overture": str(existing)}


def test_repo_that_fails_to_clone_is_skipped_and_logged_not_raised(tmp_path):
    repo_root = tmp_path / "repos"

    repos = [
        PortfolioRepo(
            repo_id="broken",
            git_url="",
            display_name="Broken",
        )
    ]

    registry = build_repo_registry(repos, str(repo_root))

    assert registry == {}
