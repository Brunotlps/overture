def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ask_returns_not_implemented(client):
    response = client.post("/ask", json={"question": "o que é isso?"})
    assert response.status_code == 501