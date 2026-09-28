"""Versioned, deterministic checks for the local answer-quality fixture."""

from dataclasses import dataclass

from app.graph import Outcome


@dataclass(frozen=True)
class EvalTurn:
    question: str
    expected_tools: tuple[str, ...]
    expected_sources: tuple[str, ...]
    required_facts: tuple[str, ...]
    forbidden_facts: tuple[str, ...] = ()
    abstention_patterns: tuple[str, ...] = ()
    expected_outcome: Outcome = Outcome.ANSWERED


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    turns: tuple[EvalTurn, ...]


# Patterns are deliberately narrow and auditable. They catch clear misses; a
# passing pattern match is not a substitute for reviewing a model's answer.
CASES: tuple[EvalCase, ...] = (
    EvalCase(
        case_id="order-behavior",
        turns=(
            EvalTurn(
                question="How does create_order calculate the order total?",
                expected_tools=("read_file",),
                expected_sources=("src/orders.py",),
                required_facts=(r"\bprice_cents\b", r"\bquantity\b", r"\b(sum|total)\b"),
            ),
        ),
    ),
    EvalCase(
        case_id="discount-references",
        turns=(
            EvalTurn(
                question="Where is DISCOUNT_RATE used?",
                expected_tools=("grep_repo",),
                expected_sources=("src/billing.py",),
                required_facts=(r"\bDISCOUNT_RATE\b", r"\bapply_discount\b"),
            ),
        ),
    ),
    EvalCase(
        case_id="money-concept",
        turns=(
            EvalTurn(
                question="How does the service handle money?",
                expected_tools=("read_file",),
                expected_sources=("src/billing.py",),
                required_facts=(r"\b(cents|amount_cents)\b", r"\b(Ledger|ledger)\b"),
            ),
        ),
    ),
    EvalCase(
        case_id="repo-files",
        turns=(
            EvalTurn(
                question="What files are in this repository?",
                expected_tools=("list_files",),
                expected_sources=("src/orders.py", "src/billing.py"),
                required_facts=(r"src/orders\.py", r"src/billing\.py"),
            ),
        ),
    ),
    EvalCase(
        case_id="missing-provider",
        turns=(
            EvalTurn(
                question="Which named payment provider does this repository integrate with?",
                expected_tools=("read_file",),
                expected_sources=("src/billing.py",),
                required_facts=(),
                abstention_patterns=(
                    r"(not specified|not named|cannot determine|does not name|doesn't name|no specific)",
                ),
                forbidden_facts=(r"\b(Stripe|PayPal|Adyen)\b",),
            ),
        ),
    ),
    EvalCase(
        case_id="order-continuation",
        turns=(
            EvalTurn(
                question="How is an order total calculated?",
                expected_tools=("read_file",),
                expected_sources=("src/orders.py",),
                required_facts=(r"\bprice_cents\b", r"\bquantity\b"),
            ),
            EvalTurn(
                question="Which identifier does the returned Order keep?",
                expected_tools=(),
                expected_sources=("src/orders.py",),
                required_facts=(r"\border_id\b",),
            ),
        ),
    ),
)
