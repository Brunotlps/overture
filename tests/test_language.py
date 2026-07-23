from unittest.mock import patch

from langchain_core.messages import AIMessage

from app.graph import agent_decide_node, budget_exceeded_node
from app.i18n import (
    ANSWER_LANGUAGE_INSTRUCTIONS,
    BUDGET_EXCEEDED_MESSAGES,
    EMPTY_FINAL_ANSWER_MESSAGES,
)
from tests.test_graph import FakeToolCallingLLM, _initial_react_state
from tests.test_health import FakeReActLLM


def _react_state_with_language(question: str, language: str):
    state = _initial_react_state(question)
    state["language"] = language
    return state


class TestAgentDecideLanguage:
    def test_system_prompt_defaults_to_portuguese_instruction(self):
        fake_llm = FakeToolCallingLLM(AIMessage(content="resposta"))

        with patch("app.graph.get_llm", return_value=fake_llm):
            agent_decide_node(_initial_react_state("O que é este projeto?"))

        system_content = fake_llm.last_messages[0].content
        assert ANSWER_LANGUAGE_INSTRUCTIONS["pt-BR"] in system_content

    def test_system_prompt_uses_english_instruction_when_requested(self):
        fake_llm = FakeToolCallingLLM(AIMessage(content="answer"))

        with patch("app.graph.get_llm", return_value=fake_llm):
            agent_decide_node(_react_state_with_language("What is this project?", "en"))

        system_content = fake_llm.last_messages[0].content
        assert ANSWER_LANGUAGE_INSTRUCTIONS["en"] in system_content

    def test_empty_answer_fallback_is_localized(self):
        fake_llm = FakeToolCallingLLM(AIMessage(content="  "))

        with patch("app.graph.get_llm", return_value=fake_llm):
            updates = agent_decide_node(
                _react_state_with_language("What is this project?", "en")
            )

        assert updates["final_answer"] == EMPTY_FINAL_ANSWER_MESSAGES["en"]


class TestBudgetExceededLanguage:
    def _state_with_tool_calls(self, language: str | None):
        state = _initial_react_state("What is this project?")
        if language is not None:
            state["language"] = language
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"relative_path": "a"}, "id": "c1"}
                ],
            )
        )
        return state

    def test_defaults_to_portuguese_message(self):
        updates = budget_exceeded_node(self._state_with_tool_calls(None))

        assert updates["final_answer"] == BUDGET_EXCEEDED_MESSAGES["pt-BR"]

    def test_uses_english_message_when_requested(self):
        updates = budget_exceeded_node(self._state_with_tool_calls("en"))

        assert updates["final_answer"] == BUDGET_EXCEEDED_MESSAGES["en"]


class TestAskLanguageContract:
    def test_ask_accepts_language_and_passes_instruction_to_llm(
        self, client, monkeypatch
    ):
        fake_llm = FakeReActLLM([AIMessage(content="answer")])
        monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

        response = client.post(
            "/ask", json={"question": "What is this project?", "language": "en"}
        )

        assert response.status_code == 200
        system_content = fake_llm.last_messages[0].content
        assert ANSWER_LANGUAGE_INSTRUCTIONS["en"] in system_content

    def test_ask_defaults_to_portuguese_when_language_omitted(
        self, client, monkeypatch
    ):
        fake_llm = FakeReActLLM([AIMessage(content="resposta")])
        monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)

        response = client.post("/ask", json={"question": "O que é este projeto?"})

        assert response.status_code == 200
        system_content = fake_llm.last_messages[0].content
        assert ANSWER_LANGUAGE_INSTRUCTIONS["pt-BR"] in system_content

    def test_ask_rejects_unsupported_language(self, client):
        response = client.post("/ask", json={"question": "Qual é?", "language": "fr"})

        assert response.status_code == 422
