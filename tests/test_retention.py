import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from app import main
from app.config import Settings
from app.graph import build_react_graph
from app.retention import LatestCheckpointSaver, ThreadRetention


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ScriptedLLM:
    """Calls read_file once per turn, then answers; can block a chosen question."""

    def __init__(self):
        self.prompts = []
        self.block_question = None
        self.blocked = threading.Event()
        self.release = threading.Event()

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.prompts.append(list(messages))
        question = next(
            m.content for m in reversed(messages) if isinstance(m, HumanMessage)
        )
        if question == self.block_question:
            self.blocked.set()
            assert self.release.wait(5)
        if isinstance(messages[-1], HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"relative_path": "README.md"},
                        "id": f"call_{len(self.prompts)}",
                    }
                ],
            )
        return AIMessage(content=f"answer to {question}")


@pytest.fixture
def env(monkeypatch, fake_repo):
    saver = LatestCheckpointSaver()
    clock = FakeClock()
    llm = ScriptedLLM()
    monkeypatch.setattr(main, "checkpointer", saver)
    monkeypatch.setattr(main, "compiled_graph", build_react_graph(checkpointer=saver))
    monkeypatch.setattr(main, "thread_retention", ThreadRetention(saver, clock=clock))
    monkeypatch.setattr("app.main.settings.repo_path", str(fake_repo))
    monkeypatch.setattr("app.main.settings.thread_ttl_seconds", 100)
    monkeypatch.setattr("app.main.settings.max_threads", 3)
    monkeypatch.setattr("app.graph.get_llm", lambda: llm)
    return SimpleNamespace(saver=saver, clock=clock, llm=llm)


def _ask(client, question, thread_id=None, **extra):
    body = {"question": question, **extra}
    if thread_id is not None:
        body["thread_id"] = thread_id
    response = client.post("/ask", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _checkpoints(saver, thread_id):
    return saver.storage.get(thread_id, {}).get("", {})


def _stored_threads(saver):
    return {tid for tid, namespaces in saver.storage.items() if namespaces.get("")}


def test_each_thread_keeps_only_its_latest_checkpoint(client, env):
    for turn in range(12):
        _ask(client, f"turn {turn}", "long")

    [latest_id] = _checkpoints(env.saver, "long")
    thread_writes = [key for key in env.saver.writes if key[0] == "long"]
    thread_blobs = [key for key in env.saver.blobs if key[0] == "long"]
    channels = [key[2] for key in thread_blobs]

    assert all(key[2] == latest_id for key in thread_writes)
    assert len(channels) == len(set(channels))
    prompt_contents = [message.content for message in env.llm.prompts[-1]]
    assert "turn 10" in prompt_contents
    assert "answer to turn 10" in prompt_contents


def test_stored_trajectory_holds_only_the_current_turn(client, env):
    for turn in range(3):
        body = _ask(client, f"turn {turn}", "trajectory")
        assert [step["tool"] for step in body["trajectory"]] == [
            "read_file",
            "agent_decide",
        ]

    state = main.compiled_graph.get_state({"configurable": {"thread_id": "trajectory"}})
    assert [step.tool for step in state.values["trajectory"]] == [
        "read_file",
        "agent_decide",
    ]


def test_idle_thread_expires_and_restarts_under_the_same_id(client, env):
    _ask(client, "first question", "idle")
    env.clock.now += 100
    _ask(client, "other thread", "other")
    assert _checkpoints(env.saver, "idle")

    env.clock.now += 1
    _ask(client, "keeps other alive", "other")
    assert not _checkpoints(env.saver, "idle")

    body = _ask(client, "after expiry", "idle")
    prompt_contents = [message.content for message in env.llm.prompts[-2]]
    assert body["thread_id"] == "idle"
    assert "first question" not in prompt_contents
    assert "after expiry" in prompt_contents


def test_expired_thread_is_no_longer_bound_to_its_repository(
    client, env, monkeypatch, tmp_path
):
    (tmp_path / "other").mkdir()
    monkeypatch.setitem(main.repo_registry, "other", str(tmp_path / "other"))
    _ask(client, "default repository", "bound")
    assert (
        client.post(
            "/ask",
            json={"question": "other repo", "thread_id": "bound", "repo_id": "other"},
        ).status_code
        == 409
    )

    env.clock.now += 101
    body = _ask(client, "other repository", "bound", repo_id="other")

    assert body["thread_id"] == "bound"


def test_new_conversations_stay_within_max_threads(client, env):
    for i in range(10):
        _ask(client, f"question {i}")

    assert len(main.thread_retention.thread_ids()) == 3
    assert len(_stored_threads(env.saver)) == 3


def test_least_recently_used_thread_is_evicted_first(client, env):
    for thread_id in ("t1", "t2", "t3"):
        _ask(client, f"hello {thread_id}", thread_id)
    _ask(client, "touch t1", "t1")
    _ask(client, "hello t4", "t4")

    assert main.thread_retention.thread_ids() == ["t3", "t1", "t4"]
    assert _stored_threads(env.saver) == {"t1", "t3", "t4"}


def test_thread_with_request_in_flight_is_never_deleted(client, env, monkeypatch):
    monkeypatch.setattr("app.main.settings.max_threads", 1)
    _ask(client, "warm up", "slow")
    env.llm.block_question = "slow question"

    with ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(_ask, client, "slow question", "slow")
        assert env.llm.blocked.wait(5)
        # Past both the TTL and the cap: only the in-flight request protects it.
        env.clock.now += 1000
        # The app intentionally uses striped locks. A hash collision would
        # serialize these unrelated requests behind the blocked slow thread.
        slow_stripe = hash("slow") % len(main.thread_locks)
        fast_ids = [
            f"fast-{i}"
            for i in range(10)
            if hash(f"fast-{i}") % len(main.thread_locks) != slow_stripe
        ][:3]
        assert len(fast_ids) == 3
        for i, thread_id in enumerate(fast_ids):
            _ask(client, f"fast {i}", thread_id)
            assert _checkpoints(env.saver, "slow")
        env.llm.release.set()
        assert slow.result()["answer"] == "answer to slow question"

    state = main.compiled_graph.get_state({"configurable": {"thread_id": "slow"}})
    assert "warm up" in [message.content for message in state.values["messages"]]
    assert main.thread_retention.thread_ids() == ["slow"]
    assert _stored_threads(env.saver) == {"slow"}


def test_keep_latest_ignores_unknown_threads():
    LatestCheckpointSaver().keep_latest("missing")


@pytest.mark.parametrize(
    "overrides",
    [
        {"thread_ttl_seconds": 0},
        {"thread_ttl_seconds": -1},
        {"max_threads": 0},
        {"max_threads": -5},
    ],
)
def test_settings_reject_non_positive_retention_limits(overrides):
    with pytest.raises(ValidationError):
        Settings(**overrides)


def test_settings_reject_non_positive_limits_from_environment(monkeypatch):
    monkeypatch.setenv("APP_MAX_THREADS", "0")

    with pytest.raises(ValidationError):
        Settings()


def test_thread_lock_is_released_even_if_retention_cleanup_fails(
    client, env, monkeypatch
):
    def failing_keep_latest(thread_id):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(env.saver, "keep_latest", failing_keep_latest)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        client.post("/ask", json={"question": "first", "thread_id": "locked"})

    stripe = main.thread_locks[hash("locked") % len(main.thread_locks)]
    assert stripe.acquire(timeout=1)
    stripe.release()
