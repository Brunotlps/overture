def test_list_repos_requires_api_key(client):
    response = client.get("/repos", headers={"X-API-Key": ""})

    assert response.status_code == 401


def test_list_repos_returns_configured_entries_excluding_failed_clones(
    client, monkeypatch
):
    monkeypatch.setattr("app.main.repo_registry", {"overture": "/some/path"})
    monkeypatch.setattr("app.main.repo_display_names", {"overture": "Overture"})

    response = client.get("/repos")

    assert response.status_code == 200
    assert response.json() == [{"repo_id": "overture", "display_name": "Overture"}]


def test_list_repos_includes_verified_revision(client, monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr("app.main.repo_registry", {"overture": "/some/path"})
    monkeypatch.setattr("app.main.repo_display_names", {"overture": "Overture"})
    monkeypatch.setattr("app.main.repo_revisions", {"overture": revision})

    response = client.get("/repos")

    assert response.status_code == 200
    assert response.json() == [
        {"repo_id": "overture", "display_name": "Overture", "revision": revision}
    ]
