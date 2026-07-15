from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import (
    EMPTY_FINAL_ANSWER,
    AgentState,
    Category,
    agent_decide_node,
    build_graph,
    route_after_decision,
)
from app.schemas import ClassificationResult
from tests.conftest import FakeLLM


def _initial_state(question: str, target: str | None = None) -> AgentState:
    return {
        "user_input": question,
        "messages": [HumanMessage(content=question)],
        "target": target,
        "category": Category.UNKNOWN,
        "tool_output": "",
        "final_answer": "",
        "trajectory": [],
        "iterations": 0,
    }


class FakeToolCallingLLM:
    def __init__(self, response):
        self._response = response
        self.bound_tools = None
        self.invocations = 0
        self.last_messages = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations += 1
        self.last_messages = messages
        return self._response


class TestAgentDecideNode:
    def test_final_answer_is_normalized_and_recorded(self):
        response = AIMessage(content="  Inspect app/main.py.  ")
        fake_llm = FakeToolCallingLLM(response)

        with patch("app.graph.get_llm", return_value=fake_llm):
            updates = agent_decide_node(_initial_state("How does /ask work?"))

        assert updates["messages"] == [response]
        assert updates["final_answer"] == "Inspect app/main.py."
        assert updates["trajectory"][0].tool == "agent_decide"
        assert updates["trajectory"][0].output_summary == "generated final answer"
        assert fake_llm.invocations == 1
        assert fake_llm.bound_tools is not None

    def test_empty_final_answer_uses_deterministic_fallback(self):
        response = AIMessage(content="  ")
        fake_llm = FakeToolCallingLLM(response)

        with patch("app.graph.get_llm", return_value=fake_llm):
            updates = agent_decide_node(_initial_state("How does /ask work?"))

        assert updates["messages"] == [response]
        assert updates["final_answer"] == EMPTY_FINAL_ANSWER
        assert (
            updates["trajectory"][0].output_summary
            == "used fallback because LLM returned no tool calls and no final content"
        )

    def test_tool_call_updates_messages_without_final_trajectory(self):
        response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"relative_path": "app/main.py"},
                    "id": "call_1",
                }
            ],
        )
        fake_llm = FakeToolCallingLLM(response)

        with patch("app.graph.get_llm", return_value=fake_llm):
            updates = agent_decide_node(_initial_state("How does /ask work?"))

        assert updates == {"messages": [response]}


class TestRouteAfterDecision:
    def test_raises_when_no_ai_message_exists(self):
        state = _initial_state("How does /ask work?")

        with pytest.raises(ValueError, match="requires at least one AIMessage"):
            route_after_decision(state)

    def test_routes_to_finalize_when_latest_ai_message_has_no_tool_calls(self):
        state = _initial_state("How does /ask work?")
        state["messages"].append(AIMessage(content="Inspect app/main.py."))

        assert route_after_decision(state) == "finalize"

    def test_routes_to_execute_tools_when_tool_calls_fit_budget(self, monkeypatch):
        state = _initial_state("How does /ask work?")
        state["iterations"] = 1
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "app/main.py"},
                        "id": "call_1",
                    },
                    {
                        "name": "list_files",
                        "args": {},
                        "id": "call_2",
                    },
                ],
            )
        )
        monkeypatch.setattr("app.graph.settings.max_iterations", 3)

        assert route_after_decision(state) == "execute_tools"

    def test_routes_to_budget_exceeded_when_tool_calls_exceed_budget(
        self, monkeypatch
    ):
        state = _initial_state("How does /ask work?")
        state["iterations"] = 2
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "app/main.py"},
                        "id": "call_1",
                    },
                    {
                        "name": "list_files",
                        "args": {},
                        "id": "call_2",
                    },
                ],
            )
        )
        monkeypatch.setattr("app.graph.settings.max_iterations", 3)

        assert route_after_decision(state) == "budget_exceeded"


class TestGraphIntegration:
    def test_structural_question_lists_files(self, fake_repo):
        fake_llm = FakeLLM(
            structured_result=ClassificationResult(
                category=Category.STRUCTURAL,
                reasoning="asks for repository structure",
            ),
            text_response="fake answer",
        )

        with (
            patch("app.graph.get_llm", return_value=fake_llm),
            patch("app.graph.settings.repo_path", str(fake_repo)),
        ):
            final_state = build_graph().invoke(
                _initial_state("What files are in this repository?")
            )

        assert final_state["final_answer"] == "fake answer"
        assert any(step.tool == "list_files" for step in final_state["trajectory"])
        assert final_state["iterations"] == 1
        assert fake_llm.structured_invocations == 1
        assert fake_llm.text_invocations == 1

    def test_unknown_category_uses_fallback_without_calling_generation_llm(
        self, fake_repo
    ):
        fake_llm = FakeLLM(
            structured_result=ClassificationResult(
                category=Category.UNKNOWN,
                reasoning="not a repository question",
            ),
            text_response="fake answer",
        )

        with patch("app.graph.get_llm", return_value=fake_llm):
            final_state = build_graph().invoke(_initial_state("What is the weather?"))

        assert "could not classify your question" in final_state["final_answer"]
        assert final_state["final_answer"] != "fake answer"
        assert [step.tool for step in final_state["trajectory"]] == [
            "classify",
            "generate_response",
        ]
        assert final_state["iterations"] == 0
        assert fake_llm.structured_invocations == 1
        assert fake_llm.text_invocations == 0
