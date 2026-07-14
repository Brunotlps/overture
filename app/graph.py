import operator
from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import OpenAIError

from app.agent_tools import get_llm_tools, get_tool_registry
from app.config import settings
from app.schemas import Category, ClassificationResult, TrajectoryStep
from app.tools import grep_repo, list_files, read_file


class AgentState(TypedDict):
    user_input: str
    messages: Annotated[list[BaseMessage], operator.add]
    
    # Day 1 compatibility fields. These should disappear when the ReAct graph
    # replaces the deterministic category-based graph.
    target: str | None
    category: Category
    tool_output: str
    final_answer: str
    
    trajectory: Annotated[list[TrajectoryStep], operator.add]
    iterations: Annotated[int, operator.add]


class FallbackReason(str, Enum):
    UNKNOWN_CATEGORY = "unknown_category"
    MISSING_TARGET = "missing_target"


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def agent_decide_node(state: AgentState) -> dict:
    """Ask the LLM to either call tools or produce the final answer.

    TODO:
    - Which messages must be sent on the first loop iteration?
    - How should the existing message history be reused on later iterations?
    - When the AIMessage has no tool_calls, where should final_answer come from?
    """
    _llm_with_tools = get_llm().bind_tools(get_llm_tools())
    raise NotImplementedError("Implement the ReAct agent decision node")


def route_after_decision(state: AgentState) -> str:
    """Route after the LLM decides whether to answer or call tools.

    TODO:
    - How do you find the latest AIMessage in state["messages"]?
    - What is the first branch if the latest AIMessage has no tool calls?
    - How do you compare requested tool_calls with remaining iteration budget?
    """
    raise NotImplementedError("Implement post-decision routing")


def execute_tools_node(state: AgentState) -> dict:
    """Execute tool calls requested by the latest AIMessage.

    TODO:
    - How do you map each tool_call name to get_tool_registry()?
    - What should happen if the LLM requests an unknown tool name?
    - How many iterations should this node add when multiple tools are called?
    - Which details belong in ToolMessage versus trajectory?
    """
    _tool_registry = get_tool_registry()
    raise NotImplementedError("Implement tool execution")


def budget_exceeded_node(state: AgentState) -> dict:
    """Stop the graph when the requested tool batch exceeds the remaining budget.

    TODO:
    - What final_answer should be deterministic and useful to the user?
    - What trajectory step proves no additional tools were executed?
    - Should messages also record this guardrail, or is trajectory enough?
    """
    raise NotImplementedError("Implement max-iterations guardrail response")


def classify_node(state: AgentState) -> dict:
    """Classify the user's question into one of the agent categories."""
    structured_llm = get_llm().with_structured_output(ClassificationResult)

    prompt = f"""
Classify the user's question into exactly one of the categories below:

- {Category.STRUCTURAL.value}: questions about repository structure, file
  organization, directories, modules, or a high-level project overview.
- {Category.SPECIFIC_CODE.value}: questions about the contents of a specific
  file, function, class, or code snippet.
- {Category.DEPENDENCIES.value}: questions that require searching for
  references, usages, calls, imports, dependencies, or occurrences of a term
  in the repository.
- {Category.UNKNOWN.value}: questions that are not about the repository, are
  too ambiguous, or do not fit any category above.

User question:
{state["user_input"]}
""".strip()

    try:
        result = structured_llm.invoke(prompt)
    except (OpenAIError, OutputParserException) as exc:
        result = ClassificationResult(
            category=Category.UNKNOWN,
            reasoning=f"Failed to classify the question: {exc}",
        )

    return {
        "category": result.category,
        "trajectory": [
            TrajectoryStep(
                tool="classify",
                tool_input=state["user_input"],
                output_summary=f"{result.category.value}: {result.reasoning}",
            )
        ],
    }


def get_fallback_reason(state: AgentState) -> FallbackReason | None:
    if state["category"] == Category.UNKNOWN:
        return FallbackReason.UNKNOWN_CATEGORY

    needs_target = state["category"] in (Category.SPECIFIC_CODE, Category.DEPENDENCIES)
    if needs_target and state["target"] is None:
        return FallbackReason.MISSING_TARGET

    return None


def route_by_category(state: AgentState) -> str:
    if get_fallback_reason(state) is not None:
        return "generate_response"

    if state["category"] == Category.STRUCTURAL:
        return "run_list_files"
    elif state["category"] == Category.SPECIFIC_CODE:
        return "run_read_file"
    elif state["category"] == Category.DEPENDENCIES:
        return "run_grep"

    raise ValueError(f"Unsupported category for routing: {state['category']}")


def run_list_files_node(state: AgentState) -> dict:
    """Execute list_files and record the result."""
    try:
        output = list_files(settings.repo_path)
        output_str = "\n".join(output)
        summary = f"found {len(output)} files"
    except FileNotFoundError as exc:
        output_str = ""
        summary = f"failed: {exc}"
    except OSError as exc:
        output_str = ""
        summary = f"failed: filesystem error: {exc}"

    return {
        "tool_output": output_str,
        "iterations": 1,
        "trajectory": [
            TrajectoryStep(
                tool="list_files",
                tool_input=settings.repo_path,
                output_summary=summary,
            )
        ],
    }


def run_read_file_node(state: AgentState) -> dict:
    """Execute read_file and record the result."""
    relative_path = state["target"]

    try:
        output_str = read_file(settings.repo_path, relative_path)
        summary = f"read {relative_path}"
    except FileNotFoundError as exc:
        output_str = ""
        summary = f"failed: file not found: {exc}"
    except ValueError as exc:
        output_str = ""
        summary = f"failed: invalid path: {exc}"
    except OSError as exc:
        output_str = ""
        summary = f"failed: filesystem error: {exc}"

    return {
        "tool_output": output_str,
        "iterations": 1,
        "trajectory": [
            TrajectoryStep(
                tool="read_file",
                tool_input=relative_path,
                output_summary=summary,
            )
        ],
    }


def run_grep_node(state: AgentState) -> dict:
    """Execute grep_repo and record the result."""
    term = state["target"]

    try:
        output = grep_repo(settings.repo_path, term)
        output_str = "\n".join(output)
        summary = f"found {len(output)} matches"
    except FileNotFoundError as exc:
        output_str = ""
        summary = f"failed: {exc}"
    except OSError as exc:
        output_str = ""
        summary = f"failed: filesystem error: {exc}"

    return {
        "tool_output": output_str,
        "iterations": 1,
        "trajectory": [
            TrajectoryStep(
                tool="grep_repo",
                tool_input=term,
                output_summary=summary,
            )
        ],
    }


def generate_response_node(state: AgentState) -> dict:
    """Generate the final answer from tool output."""
    fallback_reason = get_fallback_reason(state)

    if fallback_reason == FallbackReason.UNKNOWN_CATEGORY:
        final_answer = (
            "I could not classify your question as a repository question. "
            "Please rephrase it to ask about the repository structure, a "
            "specific file, or code references."
        )
        summary = "used fallback response for unknown category"
    elif fallback_reason == FallbackReason.MISSING_TARGET:
        final_answer = (
            "I understood the type of repository question, but I need a "
            "target. For code questions, provide a file path. For dependency "
            "or reference questions, provide a search term."
        )
        summary = "used fallback response for missing target"
    else:
        prompt = f"""
Answer the user's repository question using only the context below.

User question:
{state["user_input"]}

Tool output:
{state["tool_output"]}
""".strip()

        try:
            response = get_llm().invoke(prompt)
            final_answer = str(response.content)
            summary = "generated response with LLM"
        except OpenAIError as exc:
            final_answer = f"Failed to generate a response: {exc}"
            summary = f"failed: {exc}"

    return {
        "final_answer": final_answer,
        "trajectory": [
            TrajectoryStep(
                tool="generate_response",
                tool_input=state["user_input"],
                output_summary=summary,
            )
        ],
    }


def build_graph():
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("run_list_files", run_list_files_node)
    graph.add_node("run_read_file", run_read_file_node)
    graph.add_node("run_grep", run_grep_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {
            "run_list_files": "run_list_files",
            "run_read_file": "run_read_file",
            "run_grep": "run_grep",
            "generate_response": "generate_response",
        },
    )

    graph.add_edge("run_list_files", "generate_response")
    graph.add_edge("run_read_file", "generate_response")
    graph.add_edge("run_grep", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()
