from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph import (
    EMPTY_FINAL_ANSWER,
    REACT_SYSTEM_PROMPT,
    Category,
    DeterministicAgentState,
    ReActAgentState,
    agent_decide_node,
    budget_exceeded_node,
    build_graph,
    build_react_graph,
    execute_tools_node,
    get_latest_ai_message,
    route_after_decision,
)
from app.schemas import ClassificationResult
from tests.conftest import FakeLLM


def _initial_deterministic_state(
    question: str, target: str | None = None
) -> DeterministicAgentState:
    return {
        "user_input": question,
        "target": target,
        "category": Category.UNKNOWN,
        "tool_output": "",
        "final_answer": "",
        "trajectory": [],
        "iterations": 0,
    }


def _initial_react_state(question: str) -> ReActAgentState:
    return {
        "user_input": question,
        "repo_path": "/unused-in-tests-with-fake-tools",
        "messages": [HumanMessage(content=question)],
        "final_answer": "",
        "outcome": None,
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


class FakeSequentialToolCallingLLM:
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


class FakeTool:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.invocations = 0
        self.last_args = None

    def invoke(self, args):
        self.invocations += 1
        self.last_args = args
        if self._error is not None:
            raise self._error
        return self._result


class TestAgentDecideNode:
    def test_final_answer_is_normalized_and_recorded(self):
        response = AIMessage(content="  Inspect app/main.py.  ")
        fake_llm = FakeToolCallingLLM(response)

        with patch("app.graph.get_llm", return_value=fake_llm):
            updates = agent_decide_node(_initial_react_state("How does /ask work?"))

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
            updates = agent_decide_node(_initial_react_state("How does /ask work?"))

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
            updates = agent_decide_node(_initial_react_state("How does /ask work?"))

        assert updates == {"messages": [response]}

    def test_llm_receives_system_prompt_without_persisting_it(self):
        response = AIMessage(content="Answer.")
        fake_llm = FakeToolCallingLLM(response)
        state = _initial_react_state("How does /ask work?")

        with patch("app.graph.get_llm", return_value=fake_llm):
            updates = agent_decide_node(state)

        assert fake_llm.last_messages is not None
        assert fake_llm.last_messages[0].content.startswith(REACT_SYSTEM_PROMPT)
        assert fake_llm.last_messages[1:] == state["messages"]
        assert updates["messages"] == [response]

    def test_system_prompt_mentions_semantic_search_when_flag_enabled(
        self, monkeypatch
    ):
        monkeypatch.setattr("app.graph.settings.semantic_search_enabled", True)
        response = AIMessage(content="Answer.")
        fake_llm = FakeToolCallingLLM(response)
        state = _initial_react_state("How does /ask work?")

        with patch("app.graph.get_llm", return_value=fake_llm):
            agent_decide_node(state)

        assert "semantic_search" in fake_llm.last_messages[0].content

    def test_system_prompt_omits_semantic_search_when_flag_disabled(self):
        response = AIMessage(content="Answer.")
        fake_llm = FakeToolCallingLLM(response)
        state = _initial_react_state("How does /ask work?")

        with patch("app.graph.get_llm", return_value=fake_llm):
            agent_decide_node(state)

        assert "semantic_search" not in fake_llm.last_messages[0].content

    def test_system_prompt_reports_remaining_tool_budget(self):
        response = AIMessage(content="Answer.")
        fake_llm = FakeToolCallingLLM(response)
        state = _initial_react_state("How does /ask work?")
        state["iterations"] = 3

        with (
            patch("app.graph.get_llm", return_value=fake_llm),
            patch("app.graph.settings") as fake_settings,
        ):
            fake_settings.max_iterations = 5
            agent_decide_node(state)

        assert fake_llm.last_messages is not None
        assert "Tool budget remaining: 2 tool call(s)" in fake_llm.last_messages[0].content


class TestGetLatestAiMessage:
    def test_returns_latest_ai_message_when_multiple_ai_messages_exist(self):
        state = _initial_react_state("How does /ask work?")
        first_ai_message = AIMessage(content="First answer.")
        latest_ai_message = AIMessage(content="Latest answer.")
        state["messages"].extend(
            [
                first_ai_message,
                HumanMessage(content="Can you inspect app/graph.py too?"),
                latest_ai_message,
            ]
        )

        assert get_latest_ai_message(state) is latest_ai_message

    def test_raises_when_no_ai_message_exists(self):
        state = _initial_react_state("How does /ask work?")

        with pytest.raises(
            ValueError, match="agent state requires at least one AIMessage"
        ):
            get_latest_ai_message(state)


class TestRouteAfterDecision:
    def test_raises_when_no_ai_message_exists(self):
        state = _initial_react_state("How does /ask work?")

        with pytest.raises(
            ValueError, match="agent state requires at least one AIMessage"
        ):
            route_after_decision(state)

    def test_routes_to_finalize_when_latest_ai_message_has_no_tool_calls(self):
        state = _initial_react_state("How does /ask work?")
        state["messages"].append(AIMessage(content="Inspect app/main.py."))

        assert route_after_decision(state) == "finalize"

    def test_routes_to_execute_tools_when_tool_calls_fit_budget(self, monkeypatch):
        state = _initial_react_state("How does /ask work?")
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
        state = _initial_react_state("How does /ask work?")
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


class TestExecuteToolsNode:
    def test_executes_known_tool_and_records_message_trajectory_and_iterations(
        self, monkeypatch
    ):
        state = _initial_react_state("Read app/main.py")
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "app/main.py"},
                        "id": "call_1",
                    }
                ],
            )
        )
        fake_tool = FakeTool(result="file contents")
        monkeypatch.setattr(
            "app.graph.get_tool_registry", lambda: {"read_file": fake_tool}
        )

        updates = execute_tools_node(state)

        assert updates["iterations"] == 1
        assert fake_tool.invocations == 1
        assert fake_tool.last_args == {
            "relative_path": "app/main.py",
            "repo_path": state["repo_path"],
        }
        assert updates["messages"] == [
            ToolMessage(content="file contents", tool_call_id="call_1")
        ]
        assert updates["trajectory"][0].tool == "read_file"
        assert updates["trajectory"][0].tool_input == '{"relative_path": "app/main.py"}'
        assert updates["trajectory"][0].output_summary == "executed successfully"

    def test_unknown_tool_returns_error_message_and_counts_iteration(
        self, monkeypatch
    ):
        state = _initial_react_state("Run missing tool")
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "missing_tool",
                        "args": {"term": "FastAPI"},
                        "id": "call_1",
                    }
                ],
            )
        )
        monkeypatch.setattr("app.graph.get_tool_registry", lambda: {})

        updates = execute_tools_node(state)

        assert updates["iterations"] == 1
        assert updates["messages"] == [
            ToolMessage(
                content="Unknown tool requested: missing_tool",
                tool_call_id="call_1",
            )
        ]
        assert updates["trajectory"][0].tool == "missing_tool"
        assert updates["trajectory"][0].tool_input == '{"term": "FastAPI"}'
        assert updates["trajectory"][0].output_summary == (
            "failed: unknown tool: missing_tool"
        )

    @pytest.mark.parametrize(
        "error",
        [
            FileNotFoundError("missing file"),
            ValueError("invalid path"),
            OSError("filesystem unavailable"),
        ],
    )
    def test_expected_tool_error_returns_error_message_and_counts_iteration(
        self, monkeypatch, error
    ):
        state = _initial_react_state("Read app/main.py")
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "app/main.py"},
                        "id": "call_1",
                    }
                ],
            )
        )
        fake_tool = FakeTool(error=error)
        monkeypatch.setattr(
            "app.graph.get_tool_registry", lambda: {"read_file": fake_tool}
        )

        updates = execute_tools_node(state)

        assert updates["iterations"] == 1
        assert updates["messages"] == [
            ToolMessage(content=f"Tool error: {error}", tool_call_id="call_1")
        ]
        assert updates["trajectory"][0].output_summary == f"failed: {error}"

    def test_unexpected_tool_error_is_not_caught(self, monkeypatch):
        state = _initial_react_state("Read app/main.py")
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "app/main.py"},
                        "id": "call_1",
                    }
                ],
            )
        )
        fake_tool = FakeTool(error=RuntimeError("programming bug"))
        monkeypatch.setattr(
            "app.graph.get_tool_registry", lambda: {"read_file": fake_tool}
        )

        with pytest.raises(RuntimeError, match="programming bug"):
            execute_tools_node(state)

    def test_executes_real_agent_tool_from_registry(self, fake_repo):
        state = _initial_react_state("Read src/main.py")
        state["repo_path"] = str(fake_repo)
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "src/main.py"},
                        "id": "call_1",
                    }
                ],
            )
        )

        updates = execute_tools_node(state)

        assert updates["iterations"] == 1
        assert "def circuit_breaker" in updates["messages"][0].content
        assert updates["messages"][0].tool_call_id == "call_1"
        assert updates["trajectory"][0].tool == "read_file"
        assert updates["trajectory"][0].tool_input == '{"relative_path": "src/main.py"}'
        assert updates["trajectory"][0].output_summary == "executed successfully"

    def test_multiple_tool_calls_each_get_message_trajectory_and_iteration(
        self, monkeypatch
    ):
        state = _initial_react_state("Inspect repo")
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_files",
                        "args": {},
                        "id": "call_1",
                    },
                    {
                        "name": "grep_repo",
                        "args": {"term": "FastAPI"},
                        "id": "call_2",
                    },
                ],
            )
        )
        list_files_tool = FakeTool(result="app/main.py")
        grep_tool = FakeTool(result="app/main.py:1: from fastapi import FastAPI")
        monkeypatch.setattr(
            "app.graph.get_tool_registry",
            lambda: {"list_files": list_files_tool, "grep_repo": grep_tool},
        )

        updates = execute_tools_node(state)

        assert updates["iterations"] == 2
        assert [message.tool_call_id for message in updates["messages"]] == [
            "call_1",
            "call_2",
        ]
        assert [step.tool for step in updates["trajectory"]] == [
            "list_files",
            "grep_repo",
        ]


class TestBudgetExceededNode:
    def test_returns_guardrail_response_without_messages_or_iterations(
        self, monkeypatch
    ):
        state = _initial_react_state("How does /ask work?")
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

        updates = budget_exceeded_node(state)

        assert "maximum number of tool calls" in updates["final_answer"]
        assert "messages" not in updates
        assert "iterations" not in updates
        step = updates["trajectory"][0]
        assert step.tool == "max_iterations_guardrail"
        assert step.tool_input == "How does /ask work?"
        assert (
            step.output_summary
            == "blocked 2 tool calls because only 1 iteration remained"
        )

    def test_summary_uses_singular_tool_call_and_zero_iterations(
        self, monkeypatch
    ):
        state = _initial_react_state("How does /ask work?")
        state["iterations"] = 3
        state["messages"].append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "app/main.py"},
                        "id": "call_1",
                    },
                ],
            )
        )
        monkeypatch.setattr("app.graph.settings.max_iterations", 3)

        updates = budget_exceeded_node(state)

        assert (
            updates["trajectory"][0].output_summary
            == "blocked 1 tool call because only 0 iterations remained"
        )

    def test_summary_uses_singular_remaining_iteration(self, monkeypatch):
        state = _initial_react_state("How does /ask work?")
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
                ],
            )
        )
        monkeypatch.setattr("app.graph.settings.max_iterations", 3)

        updates = budget_exceeded_node(state)

        assert (
            updates["trajectory"][0].output_summary
            == "blocked 1 tool call because only 1 iteration remained"
        )


class TestReactGraphIntegration:
    def test_finalize_route_ends_after_agent_decision(self):
        fake_llm = FakeSequentialToolCallingLLM(
            [AIMessage(content="Inspect app/main.py.")]
        )

        with patch("app.graph.get_llm", return_value=fake_llm):
            final_state = build_react_graph().invoke(
                _initial_react_state("How does /ask work?")
            )

        assert final_state["final_answer"] == "Inspect app/main.py."
        assert [step.tool for step in final_state["trajectory"]] == ["agent_decide"]
        assert final_state["iterations"] == 0
        assert fake_llm.invocations == 1

    def test_execute_tools_route_loops_back_to_agent_decision(self, monkeypatch):
        fake_llm = FakeSequentialToolCallingLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"relative_path": "app/main.py"},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="The endpoint is defined in app/main.py."),
            ]
        )
        fake_tool = FakeTool(result="file contents")
        monkeypatch.setattr(
            "app.graph.get_tool_registry", lambda: {"read_file": fake_tool}
        )

        with patch("app.graph.get_llm", return_value=fake_llm):
            final_state = build_react_graph().invoke(
                _initial_react_state("How does /ask work?")
            )

        assert final_state["final_answer"] == "The endpoint is defined in app/main.py."
        assert fake_llm.invocations == 2
        assert fake_tool.invocations == 1
        assert final_state["iterations"] == 1
        assert [step.tool for step in final_state["trajectory"]] == [
            "read_file",
            "agent_decide",
        ]

    def test_budget_exceeded_route_ends_without_executing_tools(
        self, monkeypatch
    ):
        fake_llm = FakeSequentialToolCallingLLM(
            [
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
            ]
        )
        fake_tool = FakeTool(result="should not run")
        monkeypatch.setattr("app.graph.settings.max_iterations", 3)
        monkeypatch.setattr(
            "app.graph.get_tool_registry",
            lambda: {"read_file": fake_tool, "list_files": fake_tool},
        )

        with patch("app.graph.get_llm", return_value=fake_llm):
            final_state = build_react_graph().invoke(
                {
                    **_initial_react_state("How does /ask work?"),
                    "iterations": 2,
                }
            )

        assert "maximum number of tool calls" in final_state["final_answer"]
        assert fake_llm.invocations == 1
        assert fake_tool.invocations == 0
        assert final_state["iterations"] == 2
        assert [step.tool for step in final_state["trajectory"]] == [
            "max_iterations_guardrail"
        ]

    def test_unknown_tool_error_loops_back_to_llm_for_recovery(self):
        fake_llm = FakeSequentialToolCallingLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "missing_tool",
                            "args": {"term": "FastAPI"},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="I could not use that tool, so I recovered."),
            ]
        )

        with patch("app.graph.get_llm", return_value=fake_llm):
            final_state = build_react_graph().invoke(
                _initial_react_state("Use an unavailable tool")
            )

        assert final_state["final_answer"] == (
            "I could not use that tool, so I recovered."
        )
        assert fake_llm.invocations == 2
        assert final_state["messages"][-2].content == (
            "Unknown tool requested: missing_tool"
        )
        assert final_state["messages"][-2].tool_call_id == "call_1"
        assert final_state["iterations"] == 1
        assert [step.tool for step in final_state["trajectory"]] == [
            "missing_tool",
            "agent_decide",
        ]


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
                _initial_deterministic_state("What files are in this repository?")
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
            final_state = build_graph().invoke(
                _initial_deterministic_state("What is the weather?")
            )

        assert "could not classify your question" in final_state["final_answer"]
        assert final_state["final_answer"] != "fake answer"
        assert [step.tool for step in final_state["trajectory"]] == [
            "classify",
            "generate_response",
        ]
        assert final_state["iterations"] == 0
        assert fake_llm.structured_invocations == 1
        assert fake_llm.text_invocations == 0
