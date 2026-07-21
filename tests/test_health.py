from langchain_core.messages import AIMessage


class FakeReActLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.bound_tools = None
        self.invocations = 0
        self.last_messages = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
        self.last_messages = messages
        if not self._responses:
            raise AssertionError("Fake LLM received more invocations than expected")
        return self._responses.pop(0)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ask_returns_successful_response(client, monkeypatch):
    fake_llm = FakeReActLLM([AIMessage(content="fake answer")])

    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    response = client.post("/ask", json={"question": "What files are in this repository?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer"
    assert body["iterations"] == 0
    assert body["trajectory"][0]["tool"] == "agent_decide"
    assert fake_llm.invocations == 1
    assert fake_llm.bound_tools is not None


def test_ask_executes_tool_call_then_returns_final_answer(
    client, fake_repo, monkeypatch
):
    fake_llm = FakeReActLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "src/main.py"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="fake answer after reading the file"),
        ]
    )
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)
    monkeypatch.setattr("app.main.settings.repo_path", str(fake_repo))

    response = client.post("/ask", json={"question": "Read src/main.py"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "fake answer after reading the file"
    assert body["iterations"] == 1
    assert [step["tool"] for step in body["trajectory"]] == [
        "read_file",
        "agent_decide",
    ]
    assert fake_llm.invocations == 2


def test_ask_rejects_invalid_request(client):
    response = client.post("/ask", json={"question": "ab"})  # abaixo do min_length=3
    assert response.status_code == 422


def test_ask_error_response_does_not_leak_exception_details(client, monkeypatch):
    def broken_llm():
        raise RuntimeError("secret internal detail: /private/path api_key=abc123")

    monkeypatch.setattr("app.graph.get_llm", broken_llm)

    response = client.post("/ask", json={"question": "Anything"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Unexpected error running the agent"
    assert "secret internal detail" not in response.text
    assert "api_key" not in response.text


def test_ask_rejects_legacy_target_field(client):
    response = client.post(
        "/ask",
        json={
            "question": "What files are in this repository?",
            "target": "app/main.py",
        },
    )

    assert response.status_code == 422
