import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from app import graph as graph_module
from app.config import settings
from app.graph import ReActAgentState, build_react_graph
from app.observability import clip, configure_logging, request_id_var
from app.portfolio import load_portfolio_repos
from app.repo import build_repo_registry, ensure_repo
from app.schemas import AskRequest, AskResponse, RepoInfo
from app.security import require_api_key
from app.summarization import build_conversation_summary

SUMMARIZATION_INSTRUCTION = (
    "Summarize the following conversation concisely, preserving facts and "
    "decisions a follow-up question might need."
)

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

compiled_graph = build_react_graph(checkpointer=MemorySaver())

repo_registry: dict[str, str] = {}
repo_display_names: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_repo(settings.repo_path, settings.repo_git_url)

    portfolio_repos = load_portfolio_repos(settings.portfolio_repos_path)
    registry = build_repo_registry(portfolio_repos, settings.repo_root)
    repo_registry.clear()
    repo_registry.update(registry)
    repo_display_names.clear()
    repo_display_names.update(
        {repo.repo_id: repo.display_name for repo in portfolio_repos if repo.repo_id in registry}
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
    "/repos", response_model=list[RepoInfo], dependencies=[Security(require_api_key)]
)
def list_repos() -> list[RepoInfo]:
    return [
        RepoInfo(repo_id=repo_id, display_name=repo_display_names[repo_id])
        for repo_id in repo_registry
    ]


@app.post("/ask", response_model=AskResponse, dependencies=[Security(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    started = time.perf_counter()

    try:
        if request.repo_id is not None:
            if request.repo_id not in repo_registry:
                raise HTTPException(
                    status_code=404, detail=f"Unknown repo_id: {request.repo_id}"
                )
            repo_path = repo_registry[request.repo_id]
        else:
            repo_path = settings.repo_path

        thread_id = request.thread_id or uuid.uuid4().hex
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        existing_state = compiled_graph.get_state(config)
        history = (
            existing_state.values.get("messages", []) if existing_state.values else []
        )
        prior_iterations = (
            existing_state.values.get("iterations", 0) if existing_state.values else 0
        )
        prior_summary = (
            existing_state.values.get("conversation_summary", "")
            if existing_state.values
            else ""
        )

        excess = len(history) - settings.max_history_messages
        if excess > 0:
            messages_to_drop = history[:excess]
            state_update: dict = {
                "messages": [
                    RemoveMessage(id=message.id) for message in messages_to_drop
                ]
            }
            try:
                state_update["conversation_summary"] = build_conversation_summary(
                    messages_to_drop, prior_summary, _summarize_fn
                )
            except Exception as exc:
                logger.warning(
                    "summarization_failed",
                    extra={"thread_id": thread_id, "error": str(exc)},
                )
            compiled_graph.update_state(config, state_update)

        initial_state: ReActAgentState = {
            "user_input": request.question,
            "repo_path": repo_path,
            "messages": [HumanMessage(content=request.question)],
            "final_answer": "",
            "outcome": None,
            "trajectory": [],
            "iterations": 0,
            "turn_start_iterations": prior_iterations,
        }

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
        logger.info(
            "ask_completed",
            extra={
                "question": clip(request.question),
                "tools_called": [step.tool for step in final_state["trajectory"]],
                "iterations": final_state["iterations"],
                "outcome": outcome.value if outcome else None,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

        return AskResponse(
            answer=final_state["final_answer"],
            trajectory=final_state["trajectory"],
            iterations=final_state["iterations"],
            thread_id=thread_id,
        )
    finally:
        request_id_var.reset(token)
