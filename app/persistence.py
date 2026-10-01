"""Optional PostgreSQL conversation persistence and cross-instance coordination."""

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from app.config import settings
from app.graph import Outcome, build_react_graph
from app.retention import (
    StorageUnavailable,
    ThreadBusy,
    ThreadNotOwned,
    ThreadRepoMismatch,
)
from app.schemas import TrajectoryStep

CREATE_CONVERSATIONS_SQL = """
CREATE TABLE IF NOT EXISTS overture_conversations (
    thread_id TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    last_used TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""
CREATE_LAST_USED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS overture_conversations_last_used_idx
    ON overture_conversations (last_used)
"""


def _lock_key(thread_id: str) -> int:
    digest = hashlib.blake2b(
        thread_id.encode(), digest_size=8, person=b"overture"
    ).digest()
    return int.from_bytes(digest, "big", signed=True)


class PostgresThreadRetention:
    """Metadata and advisory locks shared by every instance using the same database."""

    def __init__(self, pool: ConnectionPool, saver: PostgresSaver):
        self._pool = pool
        self._saver = saver
        self._guard = threading.Lock()
        self._leases: dict[str, psycopg.Connection] = {}

    def begin(self, thread_id: str, principal_id: str, repo_path: str) -> None:
        try:
            conn = self._pool.getconn()
        except (psycopg.Error, PoolTimeout) as exc:
            raise StorageUnavailable() from exc
        locked = False
        retained = False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('lock_timeout', %s, false)",
                    (f"{int(settings.ask_deadline_seconds * 1000)}ms",),
                )
                cur.execute("SELECT pg_advisory_lock(%s)", (_lock_key(thread_id),))
                locked = True
                cur.execute(
                    "SELECT principal_id, repo_path, "
                    "last_used < NOW() - (%s * INTERVAL '1 second') AS expired "
                    "FROM overture_conversations WHERE thread_id = %s",
                    (settings.thread_ttl_seconds, thread_id),
                )
                row = cur.fetchone()
                if row and row["expired"]:
                    self._delete_locked(cur, thread_id)
                    row = None
                if row:
                    if row["principal_id"] != principal_id:
                        raise ThreadNotOwned()
                    if row["repo_path"] != repo_path:
                        raise ThreadRepoMismatch()
                else:
                    # Never claim a legacy checkpoint without ownership metadata.
                    cur.execute(
                        "SELECT 1 FROM checkpoints WHERE thread_id = %s LIMIT 1",
                        (thread_id,),
                    )
                    if cur.fetchone():
                        raise StorageUnavailable()
                    cur.execute(
                        "INSERT INTO overture_conversations "
                        "(thread_id, principal_id, repo_path) VALUES (%s, %s, %s)",
                        (thread_id, principal_id, repo_path),
                    )
            with self._guard:
                self._leases[thread_id] = conn
                retained = True
            return
        except psycopg.errors.LockNotAvailable as exc:
            raise ThreadBusy() from exc
        except (psycopg.Error, PoolTimeout) as exc:
            raise StorageUnavailable() from exc
        finally:
            if not retained:
                if locked:
                    try:
                        conn.execute("SELECT pg_advisory_unlock(%s)", (_lock_key(thread_id),))
                    except psycopg.Error:
                        pass
                try:
                    conn.execute("RESET lock_timeout")
                except psycopg.Error:
                    pass
                self._pool.putconn(conn)

    def end(self, thread_id: str) -> None:
        with self._guard:
            conn = self._leases.pop(thread_id)
        error = None
        try:
            conn.execute(
                "UPDATE overture_conversations SET last_used = NOW() WHERE thread_id = %s",
                (thread_id,),
            )
        except psycopg.Error as exc:
            error = exc
        finally:
            try:
                conn.execute("SELECT pg_advisory_unlock(%s)", (_lock_key(thread_id),))
            except psycopg.Error as exc:
                error = exc
            try:
                conn.execute("RESET lock_timeout")
            except psycopg.Error as exc:
                error = exc
            self._pool.putconn(conn)
        if error is not None:
            raise StorageUnavailable() from error
        self._prune()

    def _delete_locked(self, cur: psycopg.Cursor, thread_id: str) -> None:
        # Delete checkpoint first: failure must never release ownership of live data.
        self._saver.delete_thread(thread_id)
        cur.execute("DELETE FROM overture_conversations WHERE thread_id = %s", (thread_id,))

    def _prune(self) -> None:
        """Opportunistically delete expired and excess idle threads."""
        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT thread_id FROM overture_conversations "
                    "WHERE last_used < NOW() - (%s * INTERVAL '1 second') "
                    "ORDER BY last_used LIMIT 100",
                    (settings.thread_ttl_seconds,),
                )
                expired = [row["thread_id"] for row in cur.fetchall()]
                cur.execute(
                    "SELECT thread_id FROM overture_conversations "
                    "ORDER BY last_used DESC, thread_id DESC OFFSET %s LIMIT 100",
                    (settings.max_threads,),
                )
                excess = [row["thread_id"] for row in cur.fetchall()]
                for candidate in dict.fromkeys([*expired, *excess]):
                    if candidate in self._leases:
                        continue
                    key = _lock_key(candidate)
                    cur.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (key,))
                    if not cur.fetchone()["acquired"]:
                        continue
                    try:
                        cur.execute(
                            "SELECT last_used < NOW() - (%s * INTERVAL '1 second') "
                            "AS expired FROM overture_conversations WHERE thread_id = %s",
                            (settings.thread_ttl_seconds, candidate),
                        )
                        row = cur.fetchone()
                        still_excess = False
                        if row and candidate in excess and not row["expired"]:
                            cur.execute(
                                "SELECT 1 FROM (SELECT thread_id FROM "
                                "overture_conversations ORDER BY last_used DESC, "
                                "thread_id DESC OFFSET %s) AS ranked "
                                "WHERE thread_id = %s LIMIT 1",
                                (settings.max_threads, candidate),
                            )
                            still_excess = cur.fetchone() is not None
                        if row and (row["expired"] or still_excess):
                            self._delete_locked(cur, candidate)
                    finally:
                        cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
        except (psycopg.Error, PoolTimeout) as exc:
            raise StorageUnavailable() from exc


@dataclass
class PostgresRuntime:
    graph: object
    retention: PostgresThreadRetention
    checkpointer: PostgresSaver


@contextmanager
def open_postgres_runtime(dsn: str, *, setup: bool = False) -> Iterator[PostgresRuntime]:
    """Open pooled connections for the application lifespan."""
    if not dsn:
        raise StorageUnavailable("PostgreSQL DSN is required")
    with ConnectionPool(
        conninfo=dsn,
        min_size=1,
        max_size=settings.postgres_pool_max_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        open=True,
    ) as pool:
        try:
            pool.wait(timeout=10)
        except PoolTimeout as exc:
            raise StorageUnavailable("PostgreSQL is unavailable") from exc
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=(Outcome, TrajectoryStep)
        )
        saver = PostgresSaver(pool, serde=serializer)
        if setup:
            saver.setup()
            with pool.connection() as conn:
                conn.execute(CREATE_CONVERSATIONS_SQL)
                conn.execute(CREATE_LAST_USED_INDEX_SQL)
        try:
            with pool.connection() as conn:
                conn.execute(
                    "SELECT thread_id, principal_id, repo_path, last_used "
                    "FROM overture_conversations LIMIT 0"
                )
                conn.execute("SELECT thread_id FROM checkpoints LIMIT 0")
        except (psycopg.Error, PoolTimeout) as exc:
            raise StorageUnavailable("PostgreSQL schema is unavailable") from exc
        yield PostgresRuntime(
            graph=build_react_graph(checkpointer=saver),
            retention=PostgresThreadRetention(pool, saver),
            checkpointer=saver,
        )
