from unittest.mock import patch

from app.graph import AgentState, Category, build_graph
from app.schemas import ClassificationResult
from tests.conftest import FakeLLM


def _initial_state(question: str, target: str | None = None) -> AgentState:
    return {
        "user_input": question,
        "target": target,
        "category": Category.UNKNOWN,
        "tool_output": "",
        "final_answer": "",
        "trajectory": [],
        "iterations": 0,
    }


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
