from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from app.graph import build_react_graph
from app.schemas import AskRequest, AskResponse, Category

compiled_graph = build_react_graph()

app = FastAPI(title="overture", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    initial_state = {
        "user_input": request.question,
        "messages": [HumanMessage(content=request.question)],
        "target": request.target,
        "category": Category.UNKNOWN,
        "tool_output": "",
        "final_answer": "",
        "trajectory": [],
        "iterations": 0,
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error running the agent: {exc}"
        ) from exc

    return AskResponse(
        answer=final_state["final_answer"],
        trajectory=final_state["trajectory"],
        iterations=final_state["iterations"],
    )
