from dataclasses import dataclass

from app.graph import Outcome


@dataclass
class EvalCase:
    question: str
    expected_tools: list[str]
    expected_outcome: Outcome = Outcome.ANSWERED


CASES: list[EvalCase] = [
    EvalCase(
        question="How does order creation work?",
        expected_tools=["read_file"],
    ),
    EvalCase(
        question="How does the system handle money?",
        expected_tools=["read_file"],
    ),
    EvalCase(
        question="What files are in this repository?",
        expected_tools=["list_files"],
    ),
    EvalCase(
        question="Where is DISCOUNT_RATE used?",
        expected_tools=["grep_repo"],
    ),
]
