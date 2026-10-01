"""Bounded retention for in-memory conversations.

Each thread keeps only its latest checkpoint, and whole threads are dropped
after APP_THREAD_TTL_SECONDS without use or, beyond APP_MAX_THREADS, least
recently used first. A thread with a request in flight is never dropped.
"""

import threading
import time
from collections import Counter, OrderedDict

from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings


class ThreadNotOwned(Exception):
    """A principal attempted to use a live conversation belonging to another."""


class LatestCheckpointSaver(InMemorySaver):
    """In-memory checkpointer that can discard all but a thread's latest checkpoint."""

    def keep_latest(self, thread_id: str) -> None:
        """Drop older checkpoints, their pending writes, and unreferenced blobs.

        Call only while no graph run is active for thread_id.
        """
        for checkpoint_ns, checkpoints in list(self.storage.get(thread_id, {}).items()):
            if not checkpoints:
                continue
            latest_id = max(checkpoints)
            for checkpoint_id in [cid for cid in checkpoints if cid != latest_id]:
                del checkpoints[checkpoint_id]
                self.writes.pop((thread_id, checkpoint_ns, checkpoint_id), None)
            latest = self.serde.loads_typed(checkpoints[latest_id][0])
            versions = latest["channel_versions"]
            for key in [
                key
                for key in list(self.blobs)
                if key[0] == thread_id
                and key[1] == checkpoint_ns
                and versions.get(key[2]) != key[3]
            ]:
                del self.blobs[key]


class ThreadRetention:
    """Track thread use and delete conversations past the TTL or thread cap.

    begin/end must be called while holding the thread's request lock, so a
    thread is never deleted by its own request mid-run and a thread counted as
    active is never deleted by another request.
    """

    def __init__(self, saver: LatestCheckpointSaver, clock=time.monotonic):
        self._saver = saver
        self.clock = clock
        self._lock = threading.Lock()
        self._last_used: OrderedDict[str, float] = OrderedDict()
        self._active: Counter[str] = Counter()
        self._owners: dict[str, str] = {}

    def begin(self, thread_id: str, principal_id: str) -> None:
        """Expire idle threads, authorize or bind ownership, then mark active."""
        with self._lock:
            now = self.clock()
            expired = [
                tid
                for tid, last_used in self._last_used.items()
                if now - last_used > settings.thread_ttl_seconds
                and not self._active[tid]
            ]
            for tid in expired:
                self._delete(tid)
            owner = self._owners.get(thread_id)
            if owner is not None and owner != principal_id:
                raise ThreadNotOwned()
            self._owners[thread_id] = principal_id
            self._active[thread_id] += 1

    def end(self, thread_id: str) -> None:
        """Record use, keep only the latest checkpoint, and enforce the thread cap."""
        self._saver.keep_latest(thread_id)
        with self._lock:
            self._active[thread_id] -= 1
            if not self._active[thread_id]:
                del self._active[thread_id]
            self._last_used[thread_id] = self.clock()
            self._last_used.move_to_end(thread_id)
            overflow = len(self._last_used) - settings.max_threads
            for tid in [tid for tid in self._last_used if not self._active[tid]]:
                if overflow <= 0:
                    break
                self._delete(tid)
                overflow -= 1

    def thread_ids(self) -> list[str]:
        with self._lock:
            return list(self._last_used)

    def _delete(self, thread_id: str) -> None:
        self._last_used.pop(thread_id, None)
        self._saver.delete_thread(thread_id)
        self._owners.pop(thread_id, None)
