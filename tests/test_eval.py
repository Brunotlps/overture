"""Offline checks for the opt-in answer-quality evaluation."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.graph import Outcome
from app.schemas import TrajectoryStep
from eval.cases import EvalCase, EvalTurn
from eval.run import compare_reports, fixture_revision, run_case, score_turn, summarize


def _tool_exchange(path: str):
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"relative_path": path}, "id": "call-1"}
            ],
        ),
        ToolMessage(content="DISCOUNT_RATE = 0.1", tool_call_id="call-1"),
    ]


def test_wrong_fact_fails_even_with_correct_tool_and_source():
    turn = EvalTurn(
        question="What is the discount rate?",
        expected_tools=("read_file",),
        expected_sources=("src/billing.py",),
        required_facts=(r"\b0\.1\b",),
        forbidden_facts=(r"\b0\.2\b",),
    )
    state = {
        "final_answer": "src/billing.py sets the rate to 0.2",
        "outcome": Outcome.ANSWERED,
        "messages": _tool_exchange("src/billing.py"),
    }
    trajectory = [
        TrajectoryStep(
            tool="read_file",
            tool_input='{"relative_path":"src/billing.py"}',
            output_summary="executed successfully",
        )
    ]

    result = score_turn(turn, state, trajectory)

    assert result["checks"]["tools"]
    assert result["checks"]["retrieval"]
    assert result["checks"]["evidence"]
    assert not result["checks"]["factual"]
    assert not result["passed"]


def test_source_is_not_counted_when_tool_failed():
    turn = EvalTurn(
        question="What is the discount rate?",
        expected_tools=("read_file",),
        expected_sources=("src/billing.py",),
        required_facts=(r"0\.1",),
    )
    messages = _tool_exchange("src/billing.py")
    messages[1] = ToolMessage(content="Tool error: missing", tool_call_id="call-1")
    state = {
        "final_answer": "src/billing.py says 0.1",
        "outcome": Outcome.ANSWERED,
        "messages": messages,
    }

    result = score_turn(turn, state, [])

    assert not result["checks"]["retrieval"]


def test_missing_information_requires_abstention():
    turn = EvalTurn(
        question="Which provider?",
        expected_tools=(),
        expected_sources=(),
        required_facts=(),
        abstention_patterns=(r"not specified",),
    )
    state = {"final_answer": "It uses ExamplePay", "outcome": Outcome.ANSWERED, "messages": []}

    assert not score_turn(turn, state, [])["checks"]["factual"]
    state["final_answer"] = "The provider is not specified"
    assert score_turn(turn, state, [])["checks"]["factual"]


def test_repeated_turns_share_checkpoint_but_reset_turn_budget(monkeypatch):
    class FakeGraph:
        def __init__(self):
            self.state = {}
            self.inputs = []

        def get_state(self, _config):
            return SimpleNamespace(values=self.state)

        def invoke(self, initial_state, config):
            self.inputs.append((initial_state, config))
            answer = "first" if len(self.inputs) == 1 else "second"
            self.state = {
                "final_answer": answer,
                "outcome": Outcome.ANSWERED,
                "messages": [],
                "trajectory": [
                    TrajectoryStep(
                        tool="read_file", tool_input="{}", output_summary="ok"
                    )
                    for _ in self.inputs
                ],
                "iterations": len(self.inputs),
            }
            return self.state

    graph = FakeGraph()
    monkeypatch.setattr("eval.run.build_react_graph", lambda checkpointer: graph)
    case = EvalCase(
        case_id="continuation",
        turns=(
            EvalTurn("first?", (), (), (r"first",)),
            EvalTurn("second?", (), (), (r"second",)),
        ),
    )

    results = run_case(case, repeat=2)

    assert [result["iterations"] for result in results] == [1, 1]
    assert graph.inputs[0][0]["turn_start_iterations"] == 0
    assert graph.inputs[1][0]["turn_start_iterations"] == 1
    assert graph.inputs[0][1] == graph.inputs[1][1]
    assert all(result["passed"] for result in results)


def test_continuation_uses_real_graph_with_fake_llm(monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.responses = iter([AIMessage(content="first"), AIMessage(content="second")])

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            return next(self.responses)

    fake_llm = FakeLLM()
    monkeypatch.setattr("app.graph.get_llm", lambda: fake_llm)
    case = EvalCase(
        case_id="real-graph-continuation",
        turns=(
            EvalTurn("first?", (), (), (r"first",)),
            EvalTurn("second?", (), (), (r"second",)),
        ),
    )

    results = run_case(case, repeat=1)

    assert [result["answer"] for result in results] == ["first", "second"]
    assert all(result["passed"] for result in results)


def test_summary_and_comparison_keep_observed_counts_separate():
    results = [
        {
            "case_id": "a",
            "turn": 1,
            "outcome": "answered",
            "passed": False,
            "checks": {
                "conclusion": True,
                "factual": False,
                "evidence": True,
                "retrieval": True,
                "tools": True,
            },
        }
    ]
    summary = summarize(results)
    report = {
        "configuration": {"fixture_sha256": "same", "model": "a"},
        "summary": summary,
        "results": results,
    }
    baseline = {
        **report,
        "configuration": {"fixture_sha256": "same", "model": "b"},
    }

    assert summary["answered"] == 1
    assert summary["checks_passed"]["factual"] == 0
    assert summary["passed"] == 0
    assert "no statistical improvement claim" in compare_reports(report, baseline)[
        "note"
    ]

    with pytest.raises(ValueError, match="fixture revisions"):
        compare_reports(
            report,
            {**baseline, "configuration": {"fixture_sha256": "changed"}},
        )


def test_fixture_revision_changes_with_content(tmp_path):
    path = tmp_path / "example.py"
    path.write_text("answer = 1")
    before = fixture_revision(tmp_path)
    path.write_text("answer = 2")
    assert fixture_revision(tmp_path) != before
