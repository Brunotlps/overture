"""Bounded, process-local application metrics with no request-content labels."""

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

OUTCOMES = frozenset(
    {"answered", "empty_answer_fallback", "budget_exceeded", "error", "rejected"}
)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "overture_ask_requests_total",
            "Authenticated ask requests by bounded outcome",
            ("outcome",),
            registry=self.registry,
        )
        self.duration = Histogram(
            "overture_ask_duration_seconds",
            "Accepted ask duration in seconds",
            ("outcome",),
            buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, float("inf")),
            registry=self.registry,
        )
        self.iterations = Histogram(
            "overture_ask_iterations",
            "Tool calls per accepted ask",
            ("outcome",),
            buckets=(0, 1, 2, 3, 5, 8, 13, float("inf")),
            registry=self.registry,
        )
        for outcome in (*sorted(OUTCOMES), "other"):
            self.requests.labels(outcome)
            if outcome != "rejected":
                self.duration.labels(outcome)
                self.iterations.labels(outcome)

    def record_ask(
        self, outcome: str | None, duration_seconds: float, iterations: int
    ) -> None:
        label = outcome if outcome in OUTCOMES and outcome != "rejected" else "other"
        self.requests.labels(label).inc()
        self.duration.labels(label).observe(max(0, duration_seconds))
        count = iterations if isinstance(iterations, int) and iterations >= 0 else 0
        self.iterations.labels(label).observe(count)

    def record_rejection(self) -> None:
        self.requests.labels("rejected").inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)
