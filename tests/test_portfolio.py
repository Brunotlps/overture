from app.portfolio import load_portfolio_repos


def test_returns_empty_list_when_yaml_file_is_missing(tmp_path):
    missing_path = tmp_path / "does-not-exist.yaml"

    assert load_portfolio_repos(str(missing_path)) == []
