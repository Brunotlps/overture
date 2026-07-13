from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class TrajectoryStep(BaseModel):
    tool: str
    tool_input: str
    output_summary: str


class AskResponse(BaseModel):
    answer: str
    trajectory: list[TrajectoryStep]
    iterations: int