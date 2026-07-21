from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    thread_id: str | None = Field(
        default=None,
        max_length=100,
        description="Omit for a stateless request; reuse a previous thread_id to continue that conversation.",
    )
    repo_id: str | None = Field(
        default=None,
        description="Omit to use the deploy's default repo; set to a repo_id from GET /repos to target a curated portfolio repo.",
    )


class RepoInfo(BaseModel):
    repo_id: str
    display_name: str


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
    thread_id: str
