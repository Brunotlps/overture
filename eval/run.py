"""Local evaluation harness for agent answer quality (issue #5).

Runs a fixed set of questions against the fixture repo and prints a report
comparable across runs. Not part of the deployed app or CI: it calls a real
LLM, so it costs money and can be flaky. Run with:

    uv run python -m eval.run
"""

from pathlib import Path

from langchain_core.messages import HumanMessage

from app.graph import Outcome, ReActAgentState, build_react_graph
from eval.cases import CASES, EvalCase

FIXTURE_REPO = Path(__file__).parent / "fixture_repo"


def run_case(case: EvalCase) -> dict:
    graph = build_react_graph()

    initial_state: ReActAgentState = {
        "user_input": case.question,
        "repo_path": str(FIXTURE_REPO),
        "messages": [HumanMessage(content=case.question)],
        "final_answer": "",
        "outcome": None,
        "trajectory": [],
        "iterations": 0,
    }

    final_state = graph.invoke(initial_state)
    tools_called = [step.tool for step in final_state["trajectory"]]
    outcome = final_state["outcome"]

    return {
        "question": case.question,
        "outcome": outcome,
        "expected_outcome_met": outcome == case.expected_outcome,
        "tools_called": tools_called,
        "expected_tools_present": all(
            tool in tools_called for tool in case.expected_tools
        ),
        "iterations": final_state["iterations"],
    }


def main() -> None:
    results = [run_case(case) for case in CASES]

    answered = sum(1 for r in results if r["outcome"] == Outcome.ANSWERED)
    budget_exceeded = sum(
        1 for r in results if r["outcome"] == Outcome.BUDGET_EXCEEDED
    )
    expected_tool_hits = sum(1 for r in results if r["expected_tools_present"])
    avg_iterations = sum(r["iterations"] for r in results) / len(results)

    print(f"Ran {len(results)} case(s)\n")
    for r in results:
        ok = r["expected_tools_present"] and r["expected_outcome_met"]
        status = "OK" if ok else "MISS"
        outcome_label = r["outcome"].value if r["outcome"] else None
        print(f"[{status}] {r['question']}")
        print(
            f"    outcome={outcome_label} tools={r['tools_called']} "
            f"iterations={r['iterations']}"
        )

    print("\n--- Summary ---")
    print(f"answered rate:         {answered}/{len(results)}")
    print(f"budget_exceeded rate:  {budget_exceeded}/{len(results)}")
    print(f"expected tool present: {expected_tool_hits}/{len(results)}")
    print(f"avg iterations:        {avg_iterations:.1f}")


if __name__ == "__main__":
    main()
