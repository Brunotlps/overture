from pydantic import BaseModel, Field
from enum import Enum

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    target: str | None = Field(
        default=None,
        description="Optional operation target: a file path for specific_code or a search term for dependencies in Day 1's deterministic flow",
    )

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
