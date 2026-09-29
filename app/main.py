import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Overwrite

from app import graph as graph_module
from app.config import settings
from app.graph import (
    ReActAgentState,
    build_react_graph,
    reject_tool_calls,
    unanswered_tool_calls,
)
from app.observability import clip, configure_logging, request_id_var
from app.portfolio import load_portfolio_repos
from app.repo import build_repo_registry, ensure_repo
from app.retention import LatestCheckpointSaver, ThreadRetention
from app.schemas import AskRequest, AskResponse, RepoInfo
from app.security import require_api_key
from app.summarization import build_conversation_summary

SUMMARIZATION_INSTRUCTION = (
    "Summarize the following conversation concisely, preserving facts and "
    "decisions a follow-up question might need."
)

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

checkpointer = LatestCheckpointSaver()
compiled_graph = build_react_graph(checkpointer=checkpointer)
thread_retention = ThreadRetention(checkpointer)

repo_registry: dict[str, str] = {}
repo_display_names: dict[str, str] = {}
repo_revisions: dict[str, str] = {}
# Fixed stripes bound lock memory while serializing all requests for a thread.
thread_locks = tuple(threading.Lock() for _ in range(64))


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_repo(settings.repo_path, settings.repo_git_url)

    portfolio_repos = load_portfolio_repos(settings.portfolio_repos_path)
    registry = build_repo_registry(portfolio_repos, settings.repo_root)
    repo_registry.clear()
    repo_registry.update(registry)
    repo_display_names.clear()
    repo_display_names.update(
        {
            repo.repo_id: repo.display_name
            for repo in portfolio_repos
            if repo.repo_id in registry
        }
    )
    repo_revisions.clear()
    repo_revisions.update(
        {
            repo.repo_id: repo.revision
            for repo in portfolio_repos
            if repo.repo_id in registry and repo.revision is not None
        }
    )

    yield


app = FastAPI(title="overture", version="0.1.0", lifespan=lifespan)


def _summarize_fn(transcript: str) -> str:
    response = graph_module.get_llm().invoke(
        [
            SystemMessage(content=SUMMARIZATION_INSTRUCTION),
            HumanMessage(content=transcript),
        ]
    )
    return str(response.content).strip()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get(
    "/repos",
    response_model=list[RepoInfo],
    response_model_exclude_none=True,
    dependencies=[Security(require_api_key)],
)
def list_repos() -> list[RepoInfo]:
    return [
        RepoInfo(
            repo_id=repo_id,
            display_name=repo_display_names[repo_id],
            revision=repo_revisions.get(repo_id),
        )
        for repo_id in repo_registry
    ]


@app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    dependencies=[Security(require_api_key)],
)
def ask(request: AskRequest) -> AskResponse:
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    thread_lock = None
    retained_thread_id = None

    try:
        if request.repo_id is not None:
            if request.repo_id not in repo_registry:
                raise HTTPException(
                    status_code=404, detail=f"Unknown repo_id: {request.repo_id}"
                )
            repo_path = repo_registry[request.repo_id]
        else:
            repo_path = settings.repo_path
        repo_path = str(Path(repo_path).resolve())

        thread_id = request.thread_id or uuid.uuid4().hex
        thread_lock = thread_locks[hash(thread_id) % len(thread_locks)]
        thread_lock.acquire()
        thread_retention.begin(thread_id)
        retained_thread_id = thread_id
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        existing_state = compiled_graph.get_state(config)
        prior_repo_path = (
            existing_state.values.get("repo_path")
            if isinstance(existing_state.values, dict)
            else None
        )
        if prior_repo_path and str(Path(prior_repo_path).resolve()) != repo_path:
            raise HTTPException(
                status_code=409,
                detail="This thread belongs to another repository; start a new conversation.",
            )
        history = (
            existing_state.values.get("messages", []) if existing_state.values else []
        )
        # A turn that crashed mid-tool-batch leaves calls without results;
        # close them so the next prompt stays valid for the provider.
        pending_tool_calls = unanswered_tool_calls(history)
        if pending_tool_calls:
            compiled_graph.update_state(
                config,
                {
                    "messages": reject_tool_calls(
                        pending_tool_calls, "the previous request failed"
                    )
                },
            )
            history = compiled_graph.get_state(config).values["messages"]
        prior_iterations = (
            existing_state.values.get("iterations", 0) if existing_state.values else 0
        )
        prior_summary = (
            existing_state.values.get("conversation_summary", "")
            if existing_state.values
            else ""
        )

        excess = len(history) - settings.max_history_messages
        removals: list[RemoveMessage] = []
        conversation_summary = None
        if excess > 0:
            # Drop whole turns only: cutting mid-turn could split a tool call
            # from its results. With no later turn boundary, drop everything.
            cut = next(
                (
                    index
                    for index in range(excess, len(history))
                    if isinstance(history[index], HumanMessage)
                ),
                len(history),
            )
            messages_to_drop = history[:cut]
            removals = [RemoveMessage(id=message.id) for message in messages_to_drop]
            try:
                conversation_summary = build_conversation_summary(
                    messages_to_drop, prior_summary, _summarize_fn
                )
            except Exception as exc:  # noqa: BLE001 - summary failure must not block the request
                logger.warning(
                    "summarization_failed",
                    extra={"thread_id": thread_id, "error": str(exc)},
                )

        # Removals ride along with the new turn's input: a separate update_state
        # would re-run agent_decide's routing on a possibly emptied history.
        initial_state: ReActAgentState = {
            "user_input": request.question,
            "repo_path": repo_path,
            "language": request.language,
            "messages": [*removals, HumanMessage(content=request.question)],
            "final_answer": "",
            "outcome": None,
            # Trajectory is per turn; overwriting avoids accumulating it forever.
            "trajectory": Overwrite([]),
            "iterations": 0,
            "turn_start_iterations": prior_iterations,
        }
        if conversation_summary is not None:
            initial_state["conversation_summary"] = conversation_summary

        try:
            final_state = compiled_graph.invoke(initial_state, config=config)
        except Exception as exc:
            logger.exception(
                "ask_failed",
                extra={
                    "question": clip(request.question),
                    "error": str(exc),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            raise HTTPException(
                status_code=500, detail="Unexpected error running the agent"
            ) from exc

        outcome = final_state.get("outcome")
        turn_trajectory = final_state["trajectory"]
        turn_iterations = final_state["iterations"] - prior_iterations
        logger.info(
            "ask_completed",
            extra={
                "question": clip(request.question),
                "tools_called": [step.tool for step in turn_trajectory],
                "iterations": turn_iterations,
                "outcome": outcome.value if outcome else None,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

        return AskResponse(
            answer=final_state["final_answer"],
            trajectory=turn_trajectory,
            iterations=turn_iterations,
            thread_id=thread_id,
            repo_revision=repo_revisions.get(request.repo_id) if request.repo_id else None,
        )
    finally:
        try:
            if retained_thread_id is not None:
                thread_retention.end(retained_thread_id)
        finally:
            if thread_lock is not None:
                thread_lock.release()
            request_id_var.reset(token)
