from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from tests.conftest import TEST_API_KEY

QUESTION = {"question": "What files exist?"}


class TestAskAuthentication:
    def test_missing_key_returns_401_without_running_the_graph(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", TEST_API_KEY)

        with patch("app.main.compiled_graph") as fake_graph:
            response = TestClient(app).post("/ask", json=QUESTION)

        assert response.status_code == 401
        fake_graph.invoke.assert_not_called()

    def test_wrong_key_returns_401_without_running_the_graph(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", TEST_API_KEY)

        with patch("app.main.compiled_graph") as fake_graph:
            response = TestClient(app).post(
                "/ask", json=QUESTION, headers={"X-API-Key": "wrong-key"}
            )

        assert response.status_code == 401
        fake_graph.invoke.assert_not_called()

    def test_unconfigured_server_fails_closed_with_503(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "")

        with patch("app.main.compiled_graph") as fake_graph:
            response = TestClient(app).post(
                "/ask", json=QUESTION, headers={"X-API-Key": "any-key"}
            )

        assert response.status_code == 503
        fake_graph.invoke.assert_not_called()

    def test_valid_key_reaches_the_graph(self, client):
        final_state = {
            "final_answer": "ok",
            "outcome": None,
            "trajectory": [],
            "iterations": 0,
        }

        with patch("app.main.compiled_graph") as fake_graph:
            fake_graph.invoke.return_value = final_state
            response = client.post("/ask", json=QUESTION)

        assert response.status_code == 200
        fake_graph.invoke.assert_called_once()

    def test_health_stays_public(self):
        response = TestClient(app).get("/health")

        assert response.status_code == 200
