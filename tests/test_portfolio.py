import pytest

from app.portfolio import PortfolioRepo, load_portfolio_repos


def test_returns_empty_list_when_yaml_file_is_missing(tmp_path):
    missing_path = tmp_path / "does-not-exist.yaml"

    assert load_portfolio_repos(str(missing_path)) == []


def test_parses_valid_yaml_into_portfolio_repos(tmp_path):
    yaml_path = tmp_path / "portfolio_repos.yaml"
    yaml_path.write_text(
        """
        repos:
          - repo_id: overture
            git_url: https://github.com/Brunotlps/overture
            display_name: "Overture"
          - repo_id: other-project
            git_url: https://github.com/Brunotlps/other-project
            display_name: "Other Project"
        """
    )

    result = load_portfolio_repos(str(yaml_path))

    assert result == [
        PortfolioRepo(
            repo_id="overture",
            git_url="https://github.com/Brunotlps/overture",
            display_name="Overture",
        ),
        PortfolioRepo(
            repo_id="other-project",
            git_url="https://github.com/Brunotlps/other-project",
            display_name="Other Project",
        ),
    ]


def test_raises_on_malformed_yaml(tmp_path):
    yaml_path = tmp_path / "portfolio_repos.yaml"
    yaml_path.write_text("repos: [this is not: valid: yaml")

    with pytest.raises(ValueError):
        load_portfolio_repos(str(yaml_path))


def test_raises_on_entry_missing_required_field(tmp_path):
    yaml_path = tmp_path / "portfolio_repos.yaml"
    yaml_path.write_text(
        """
        repos:
          - repo_id: overture
            display_name: "Overture"
        """
    )

    with pytest.raises(ValueError):
        load_portfolio_repos(str(yaml_path))


def test_raises_on_duplicate_repo_id(tmp_path):
    yaml_path = tmp_path / "portfolio_repos.yaml"
    yaml_path.write_text(
        """
        repos:
          - repo_id: overture
            git_url: https://github.com/Brunotlps/overture
            display_name: "Overture"
          - repo_id: overture
            git_url: https://github.com/Brunotlps/overture-other-url
            display_name: "Overture again"
        """
    )

    with pytest.raises(ValueError):
        load_portfolio_repos(str(yaml_path))
