from fastapi import FastAPI, HTTPException

from app.schemas import AskRequest, AskResponse

app = FastAPI(title="overture", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    raise HTTPException(status_code=501, detail="Agent not implemented yet")