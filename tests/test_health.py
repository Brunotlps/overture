from app.schemas import Category, ClassificationResult
from tests.conftest import FakeLLM


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ask_returns_successful_response(client, fake_repo, monkeypatch):
    fake_llm = FakeLLM(
        structured_result=ClassificationResult(
            category=Category.STRUCTURAL,
            reasoning="asks for repository structure",
        ),
        text_response="fake answer",
    )

    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)
    monkeypatch.setattr("app.graph.settings.repo_path", str(fake_repo))

    response = client.post("/ask", json={"question": "What files are in this repository?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["iterations"] == 1
    assert any(step["tool"] == "list_files" for step in body["trajectory"])


def test_ask_rejects_invalid_request(client):
    response = client.post("/ask", json={"question": "ab"})  # abaixo do min_length=3
    assert response.status_code == 422