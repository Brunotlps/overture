"""Opt-in answer-quality evaluation against a fixed local repository.

This module calls a real LLM only from ``main`` / ``run_case``. Pure scoring
functions are exercised offline by pytest. Run with ``uv run python -m eval.run``.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.graph import (
    REACT_SYSTEM_PROMPT,
    SEMANTIC_SEARCH_PROMPT_ADDENDUM,
    Outcome,
    ReActAgentState,
    build_react_graph,
)
from eval.cases import CASES, EvalCase, EvalTurn

FIXTURE_REPO = Path(__file__).parent / "fixture_repo"
CHECK_NAMES = ("conclusion", "factual", "evidence", "retrieval", "tools")


def fixture_revision(root: Path = FIXTURE_REPO) -> str:
    """Hash paths and bytes so the fixture revision works without a Git repo."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sources_seen(messages: list, expected_sources: tuple[str, ...]) -> list[str]:
    """Find expected paths actually returned by repository tools."""
    calls = {}
    found = set()
    for message in messages:
        if isinstance(message, AIMessage):
            calls.update(
                (call["id"], (call["name"], call["args"]))
                for call in message.tool_calls
            )
        elif isinstance(message, ToolMessage):
            call = calls.get(message.tool_call_id)
            if call is None or str(message.content).startswith("Tool error:"):
                continue
            name, args = call
            for source in expected_sources:
                read_hit = name == "read_file" and args.get("relative_path") == source
                listing_hit = (
                    name in {"grep_repo", "list_files", "semantic_search"}
                    and source in str(message.content)
                )
                if read_hit or listing_hit:
                    found.add(source)
    return sorted(found)


def score_turn(
    turn: EvalTurn,
    final_state: ReActAgentState,
    turn_trajectory: list,
) -> dict:
    """Score observable facts separately; no check is presented as accuracy."""
    answer = final_state["final_answer"]
    observed_sources = sources_seen(final_state["messages"], turn.expected_sources)
    called_tools = [step.tool for step in turn_trajectory]
    missing_facts = [
        pattern
        for pattern in turn.required_facts
        if re.search(pattern, answer, flags=re.IGNORECASE) is None
    ]
    forbidden_facts = [
        pattern
        for pattern in turn.forbidden_facts
        if re.search(pattern, answer, flags=re.IGNORECASE) is not None
    ]
    abstention_met = (
        any(re.search(pattern, answer, flags=re.IGNORECASE) for pattern in turn.abstention_patterns)
        if turn.abstention_patterns
        else None
    )
    outcome = final_state["outcome"]
    checks = {
        "conclusion": outcome == turn.expected_outcome,
        "factual": not missing_facts and not forbidden_facts and abstention_met is not False,
        "evidence": all(source in answer for source in turn.expected_sources),
        "retrieval": all(source in observed_sources for source in turn.expected_sources),
        "tools": all(tool in called_tools for tool in turn.expected_tools),
    }
    return {
        "question": turn.question,
        "answer": answer,
        "outcome": outcome.value if isinstance(outcome, Outcome) else outcome,
        "expected_outcome": turn.expected_outcome.value,
        "tools_called": called_tools,
        "expected_tools": list(turn.expected_tools),
        "expected_sources": list(turn.expected_sources),
        "sources_seen": observed_sources,
        "required_facts": list(turn.required_facts),
        "missing_facts": missing_facts,
        "forbidden_facts_found": forbidden_facts,
        "abstention_expected": bool(turn.abstention_patterns),
        "abstention_met": abstention_met,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_case(case: EvalCase, repeat: int) -> list[dict]:
    """Run all turns on one checkpointed graph, then discard it for the next run."""
    graph = build_react_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": f"{case.case_id}-{repeat}"}}
    results = []
    for turn_number, turn in enumerate(case.turns, start=1):
        prior_state = graph.get_state(config).values or {}
        prior_iterations = prior_state.get("iterations", 0)
        prior_trajectory_length = len(prior_state.get("trajectory", []))
        initial_state: ReActAgentState = {
            "user_input": turn.question,
            "repo_path": str(FIXTURE_REPO),
            "messages": [HumanMessage(content=turn.question)],
            "final_answer": "",
            "outcome": None,
            "trajectory": [],
            "iterations": 0,
            "turn_start_iterations": prior_iterations,
        }
        final_state = graph.invoke(initial_state, config=config)
        result = score_turn(
            turn, final_state, final_state["trajectory"][prior_trajectory_length:]
        )
        result.update(case_id=case.case_id, turn=turn_number, repeat=repeat)
        result["iterations"] = final_state["iterations"] - prior_iterations
        results.append(result)
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    return {
        "turns": total,
        "answered": sum(r["outcome"] == Outcome.ANSWERED.value for r in results),
        "passed": sum(r["passed"] for r in results),
        "checks_passed": {
            name: sum(r["checks"][name] for r in results) for name in CHECK_NAMES
        },
    }


def build_report(results: list[dict], repeats: int) -> dict:
    prompt = REACT_SYSTEM_PROMPT
    if settings.semantic_search_enabled:
        prompt += SEMANTIC_SEARCH_PROMPT_ADDENDUM
    return {
        "schema_version": 1,
        "configuration": {
            "model": settings.llm_model,
            "semantic_search_enabled": settings.semantic_search_enabled,
            "max_iterations": settings.max_iterations,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "fixture_sha256": fixture_revision(),
            "repeats": repeats,
        },
        "summary": summarize(results),
        "results": results,
    }


def compare_reports(current: dict, baseline: dict) -> dict:
    """Return observed counts only; repeated model calls are not independent trials."""
    if current["configuration"]["fixture_sha256"] != baseline["configuration"][
        "fixture_sha256"
    ]:
        raise ValueError("Cannot compare reports from different fixture revisions")
    current_cases = {(r["case_id"], r["turn"]) for r in current["results"]}
    baseline_cases = {(r["case_id"], r["turn"]) for r in baseline["results"]}
    if current_cases != baseline_cases:
        raise ValueError("Cannot compare reports with different case sets")
    return {
        "baseline_configuration": baseline["configuration"],
        "baseline_summary": baseline["summary"],
        "current_configuration": current["configuration"],
        "current_summary": current["summary"],
        "note": "Observed counts only; no statistical improvement claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Write the full JSON report")
    parser.add_argument("--compare", type=Path, help="Compare with a saved JSON report")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    results = [
        result
        for repeat in range(1, args.repeats + 1)
        for case in CASES
        for result in run_case(case, repeat)
    ]
    report = build_report(results, args.repeats)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(report["summary"], indent=2))
    if args.compare:
        baseline = json.loads(args.compare.read_text())
        print(json.dumps(compare_reports(report, baseline), indent=2))


if __name__ == "__main__":
    main()
