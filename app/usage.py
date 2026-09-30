"""In-process admission for authenticated /ask requests.

The lease covers the entire request, including summarization and embeddings.
Rate counts admitted requests; concurrency counts active requests.
"""

import math
import threading
import time
from collections import defaultdict, deque
from contextlib import AbstractContextManager
from contextvars import ContextVar

request_deadline: ContextVar[float | None] = ContextVar(
    "request_deadline", default=None
)


class RequestDeadlineExceeded(TimeoutError):
    pass


class ModelInputTooLarge(ValueError):
    pass


def remaining_provider_timeout(configured_timeout: float) -> float:
    deadline = request_deadline.get()
    if deadline is None:
        return configured_timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RequestDeadlineExceeded("Request deadline exceeded")
    return min(configured_timeout, remaining)


class QuotaExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = max(1, retry_after)
        super().__init__("Request quota exceeded")


class AdmissionLease(AbstractContextManager):
    def __init__(self, controller: "AdmissionController", client_id: str):
        self.controller = controller
        self.client_id = client_id
        self.closed = False

    def close(self) -> None:
        with self.controller.lock:
            if self.closed:
                return
            self.controller.active_global -= 1
            self.controller.active_clients[self.client_id] -= 1
            if self.controller.active_clients[self.client_id] == 0:
                del self.controller.active_clients[self.client_id]
            self.closed = True

    def __exit__(self, *_exc) -> None:
        self.close()


class AdmissionController:
    def __init__(
        self,
        *,
        per_client_rate: int,
        global_rate: int,
        window_seconds: int,
        per_client_concurrency: int,
        global_concurrency: int,
    ):
        self.per_client_rate = per_client_rate
        self.global_rate = global_rate
        self.window_seconds = window_seconds
        self.per_client_concurrency = per_client_concurrency
        self.global_concurrency = global_concurrency
        self.lock = threading.Lock()
        self.recent_global: deque[float] = deque()
        self.recent_clients: dict[str, deque[float]] = defaultdict(deque)
        self.active_global = 0
        self.active_clients: dict[str, int] = defaultdict(int)

    def acquire(self, client_id: str, *, now: float | None = None) -> AdmissionLease:
        now = time.monotonic() if now is None else now
        with self.lock:
            cutoff = now - self.window_seconds
            while self.recent_global and self.recent_global[0] <= cutoff:
                self.recent_global.popleft()
            for key, recent in list(self.recent_clients.items()):
                while recent and recent[0] <= cutoff:
                    recent.popleft()
                if not recent:
                    del self.recent_clients[key]
            client_recent = self.recent_clients[client_id]
            retry = [
                math.ceil(self.recent_global[0] + self.window_seconds - now)
                if len(self.recent_global) >= self.global_rate
                else 0,
                math.ceil(client_recent[0] + self.window_seconds - now)
                if len(client_recent) >= self.per_client_rate
                else 0,
            ]
            if (
                self.active_global >= self.global_concurrency
                or self.active_clients.get(client_id, 0) >= self.per_client_concurrency
            ):
                retry.append(1)
            if any(retry):
                raise QuotaExceeded(max(retry))
            self.recent_global.append(now)
            client_recent.append(now)
            self.active_global += 1
            self.active_clients[client_id] += 1
            return AdmissionLease(self, client_id)
