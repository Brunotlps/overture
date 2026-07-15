from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)


class Category(str, Enum):
    STRUCTURAL = "structural"
    SPECIFIC_CODE = "specific_code"
    DEPENDENCIES = "dependencies"
    UNKNOWN = "unknown"


class ClassificationResult(BaseModel):
    category: Category
    reasoning: str = Field(description="Brief justification for the chosen category")


class TrajectoryStep(BaseModel):
    tool: str
    tool_input: str
    output_summary: str


class AskResponse(BaseModel):
    answer: str
    trajectory: list[TrajectoryStep]
    iterations: int
