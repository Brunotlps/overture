from langchain_core.messages import AIMessage

from tests.test_health import FakeReActLLM


def test_ask_without_thread_id_returns_a_fresh_one_each_time(client, monkeypatch):
    fake_llm = FakeReActLLM(
        [AIMessage(content="answer one"), AIMessage(content="answer two")]
    )
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    first = client.post("/ask", json={"question": "What does this service do?"})
    second = client.post("/ask", json={"question": "What does this service do?"})

    assert first.json()["thread_id"] != second.json()["thread_id"]


def test_reusing_thread_id_preserves_conversation_history(client, monkeypatch):
    fake_llm = FakeReActLLM(
        [AIMessage(content="first answer"), AIMessage(content="second answer")]
    )
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    first = client.post(
        "/ask",
        json={"question": "What does this service do?", "thread_id": "conv-1"},
    )
    assert first.status_code == 200
    assert first.json()["thread_id"] == "conv-1"

    second = client.post(
        "/ask", json={"question": "Can you elaborate?", "thread_id": "conv-1"}
    )
    assert second.status_code == 200
    assert second.json()["answer"] == "second answer"

    assert fake_llm.last_messages is not None
    history_contents = [message.content for message in fake_llm.last_messages]
    assert "What does this service do?" in history_contents
    assert "first answer" in history_contents
    assert "Can you elaborate?" in history_contents


def test_tool_budget_resets_on_new_turn_same_thread(client, fake_repo, monkeypatch):
    monkeypatch.setattr("app.graph.settings.max_iterations", 1)
    monkeypatch.setattr("app.main.settings.repo_path", str(fake_repo))
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
            AIMessage(content="answer after first tool call"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "src/utils.py"},
                        "id": "call_2",
                    }
                ],
            ),
            AIMessage(content="answer after second tool call"),
        ]
    )
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    first = client.post(
        "/ask", json={"question": "Read src/main.py", "thread_id": "budget-1"}
    )
    assert first.status_code == 200
    assert first.json()["answer"] == "answer after first tool call"

    second = client.post(
        "/ask", json={"question": "Now read src/utils.py", "thread_id": "budget-1"}
    )
    assert second.status_code == 200
    assert second.json()["answer"] == "answer after second tool call"


def test_history_beyond_limit_is_summarized_instead_of_dropped(client, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_history_messages", 2)
    fake_llm = FakeReActLLM(
        [
            AIMessage(content="answer 1"),
            AIMessage(content="answer 2"),
            AIMessage(content="Summary: discussed question 1 and 2."),
            AIMessage(content="answer 3"),
        ]
    )
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    thread_id = "sum-1"
    for i in range(1, 4):
        response = client.post(
            "/ask", json={"question": f"question {i}", "thread_id": thread_id}
        )
        assert response.status_code == 200

    from app.main import compiled_graph

    state = compiled_graph.get_state({"configurable": {"thread_id": thread_id}})
    contents = [message.content for message in state.values["messages"]]

    assert contents == ["question 2", "answer 2", "question 3", "answer 3"]
    assert state.values["conversation_summary"] == "Summary: discussed question 1 and 2."


def test_conversation_summary_accumulates_across_multiple_truncations(
    client, monkeypatch
):
    monkeypatch.setattr("app.main.settings.max_history_messages", 2)
    prior_summaries_seen = []

    def fake_build_conversation_summary(messages, prior_summary, summarize_fn):
        prior_summaries_seen.append(prior_summary)
        return f"summary-{len(prior_summaries_seen)}"

    monkeypatch.setattr(
        "app.main.build_conversation_summary", fake_build_conversation_summary
    )
    fake_llm = FakeReActLLM([AIMessage(content=f"answer {i}") for i in range(1, 6)])
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

    thread_id = "accum-1"
    for i in range(1, 6):
        response = client.post(
            "/ask", json={"question": f"question {i}", "thread_id": thread_id}
        )
        assert response.status_code == 200

    # max_history_messages=2: 2 remaining + 2 new messages exceeds the limit
    # again every turn from turn 3 onward, so it triggers on turns 3, 4, 5.
    assert prior_summaries_seen == ["", "summary-1", "summary-2"]
