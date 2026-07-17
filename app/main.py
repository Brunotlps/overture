import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security
from langchain_core.messages import HumanMessage

from app.config import settings
from app.graph import ReActAgentState, build_react_graph
from app.observability import clip, configure_logging, request_id_var
from app.repo import ensure_repo
from app.schemas import AskRequest, AskResponse
from app.security import require_api_key

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

compiled_graph = build_react_graph()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_repo(settings.repo_path, settings.repo_git_url)
    yield


app = FastAPI(title="overture", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/ask", response_model=AskResponse, dependencies=[Security(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    request_id = uuid.uuid4().hex
    token = request_id_var.set(request_id)
    started = time.perf_counter()

    initial_state: ReActAgentState = {
        "user_input": request.question,
        "messages": [HumanMessage(content=request.question)],
        "final_answer": "",
        "outcome": None,
        "trajectory": [],
        "iterations": 0,
    }

    try:
        try:
            final_state = compiled_graph.invoke(initial_state)
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
        )
    finally:
        request_id_var.reset(token)
