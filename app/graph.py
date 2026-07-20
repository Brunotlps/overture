import json
import logging
import operator
import time
from enum import Enum
from typing import Annotated, NotRequired, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAIError

from app.agent_tools import get_llm_tools, get_tool_registry
from app.config import settings
from app.observability import clip
from app.schemas import Category, ClassificationResult, TrajectoryStep
from app.tools import grep_repo, list_files, read_file

REACT_SYSTEM_PROMPT = """
You are a code question-answering agent for a single repository. You can call
these tools:

- list_files: list the repository files.
- read_file: read the full content of one file by relative path.
- grep_repo: search for a term and get the matching lines with file paths.

Investigation strategy:

1. If the question asks what something IS, how it WORKS, or how it BEHAVES
   (a function, class, endpoint, config value, feature), you must read the
   file that implements it before answering. grep_repo output only shows
   isolated matching lines — use it to LOCATE the definition, then call
   read_file on that file to understand the actual implementation.
2. Only answer directly from grep_repo results when the question is purely
   about where or how often a term appears (references, usages, imports).
3. If you do not know which file is relevant, start with grep_repo for a
   distinctive term from the question, or list_files for structural or
   overview questions.
4. grep_repo does exact substring matching, so when a search misses the
   implementation, retry with a naming variant: camelCase, PascalCase,
   snake_case, or the words joined or split (e.g. "circuit breaker" vs
   "CircuitBreaker" vs "circuitbreaker"). Tool calls are limited, so try at
   most two variants; if they also miss, call list_files instead — file paths
   are strong hints for where a concept is implemented.
5. If a tool result does not contain what you need, try a different term or
   file instead of guessing.

Answering rules:

- Ground every claim in file content you actually read in this conversation;
  never answer from assumptions about what the code probably does.
- Cite the relevant file paths in the answer.
- If the repository does not contain the information, say so explicitly.
- Tool calls are limited, so make each one purposeful; when you already have
  enough context, answer instead of calling more tools.
""".strip()

logger = logging.getLogger(__name__)

EMPTY_FINAL_ANSWER = (
    "I could not produce a final answer from the model response. "
    "Please try rephrasing the question."
)


class DeterministicAgentState(TypedDict):
    user_input: str
    target: str | None
    category: Category
    tool_output: str
    final_answer: str
    trajectory: Annotated[list[TrajectoryStep], operator.add]
    iterations: Annotated[int, operator.add]


class Outcome(str, Enum):
    ANSWERED = "answered"
    EMPTY_ANSWER_FALLBACK = "empty_answer_fallback"
    BUDGET_EXCEEDED = "budget_exceeded"


class ReActAgentState(TypedDict):
    user_input: str
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str
    outcome: Outcome | None
    trajectory: Annotated[list[TrajectoryStep], operator.add]
    iterations: Annotated[int, operator.add]
    # Cumulative iteration count carried over from prior turns on the same
    # thread_id, so the per-question tool budget resets each turn instead of
    # shrinking across an entire conversation. Absent for single-turn/no
    # checkpointer use, hence the .get(..., 0) reads below.
    turn_start_iterations: NotRequired[int]


class FallbackReason(str, Enum):
    UNKNOWN_CATEGORY = "unknown_category"
    MISSING_TARGET = "missing_target"


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


def agent_decide_node(state: ReActAgentState) -> dict:
    """Ask the LLM to either call tools or produce the final answer.

    Tool requests are recorded only in messages. Trajectory is updated here
    only when the model produces a final answer or an empty-answer fallback.
    """
    llm_with_tools = get_llm().bind_tools(get_llm_tools())
    turn_iterations = state["iterations"] - state.get("turn_start_iterations", 0)
    remaining_budget = settings.max_iterations - turn_iterations
    system_content = (
        f"{REACT_SYSTEM_PROMPT}\n\n"
        f"Tool budget remaining: {remaining_budget} tool call(s). Requests beyond "
        "the budget are rejected without an answer, so when the budget is nearly "
        "exhausted, stop searching and answer with the information you already have."
    )
    prompt_messages = [SystemMessage(content=system_content), *state["messages"]]
    response = llm_with_tools.invoke(prompt_messages)
    tool_calls = getattr(response, "tool_calls", []) or []

    updates: dict = {"messages": [response]}

    if tool_calls:
        return updates

    final_answer = str(response.content).strip()
    if final_answer:
        summary = "generated final answer"
        outcome = Outcome.ANSWERED
    else:
        final_answer = EMPTY_FINAL_ANSWER
        summary = (
            "used fallback because LLM returned no tool calls and no final content"
        )
        outcome = Outcome.EMPTY_ANSWER_FALLBACK

    updates["final_answer"] = final_answer
    updates["outcome"] = outcome
    updates["trajectory"] = [
        TrajectoryStep(
            tool="agent_decide",
            tool_input=state["user_input"],
            output_summary=summary,
        )
    ]
    return updates


def get_latest_ai_message(state: ReActAgentState) -> AIMessage:
    """Return the latest AI message recorded in the conversation state."""
    latest_ai_message = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, AIMessage)
        ),
        None,
    )
    if latest_ai_message is None:
        raise ValueError("agent state requires at least one AIMessage")

    return latest_ai_message


def route_after_decision(state: ReActAgentState) -> str:
    """Route after the LLM decides whether to answer or call tools."""
    latest_ai_message = get_latest_ai_message(state)
    tool_calls = latest_ai_message.tool_calls or []
    turn_iterations = state["iterations"] - state.get("turn_start_iterations", 0)
    if not tool_calls:
        route = "finalize"
    elif turn_iterations + len(tool_calls) > settings.max_iterations:
        route = "budget_exceeded"
    else:
        route = "execute_tools"

    logger.info(
        "route_selected",
        extra={
            "route": route,
            "requested_tools": [tool_call["name"] for tool_call in tool_calls],
            "iterations": state["iterations"],
        },
    )
    return route


def execute_tools_node(state: ReActAgentState) -> dict:
    """Execute tool calls requested by the latest AIMessage."""
    tool_registry = get_tool_registry()
    latest_ai_message = get_latest_ai_message(state)
    tool_messages = []
    trajectory = []

    for tool_call in latest_ai_message.tool_calls or []:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]
        serialized_input = json.dumps(tool_args, sort_keys=True)

        tool = tool_registry.get(tool_name)
        started = time.perf_counter()
        if tool is None:
            content = f"Unknown tool requested: {tool_name}"
            summary = f"failed: unknown tool: {tool_name}"
            status = "unknown_tool"
        else:
            try:
                content = str(tool.invoke(tool_args))
                summary = "executed successfully"
                status = "ok"
            except (FileNotFoundError, ValueError, OSError) as exc:
                content = f"Tool error: {exc}"
                summary = f"failed: {exc}"
                status = "error"

        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.log(
            logging.INFO if status == "ok" else logging.WARNING,
            "tool_executed",
            extra={
                "tool": tool_name,
                "tool_input": clip(serialized_input),
                "status": status,
                "duration_ms": duration_ms,
                "output_summary": summary,
            },
        )

        tool_messages.append(
            ToolMessage(content=content, tool_call_id=tool_call_id)
        )
        trajectory.append(
            TrajectoryStep(
                tool=tool_name,
                tool_input=serialized_input,
                output_summary=summary,
            )
        )

    return {
        "messages": tool_messages,
        "trajectory": trajectory,
        "iterations": len(tool_messages),
    }


def budget_exceeded_node(state: ReActAgentState) -> dict:
    """Stop the graph when the requested tool batch exceeds the remaining budget."""
    latest_ai_message = get_latest_ai_message(state)
    requested_tool_calls = len(latest_ai_message.tool_calls or [])
    turn_iterations = state["iterations"] - state.get("turn_start_iterations", 0)
    remaining_budget = settings.max_iterations - turn_iterations

    final_answer = (
        "I reached the maximum number of tool calls allowed for this request. "
        "Please narrow the question or ask about a more specific part of the "
        "repository."
    )

    tool_call_label = "tool call" if requested_tool_calls == 1 else "tool calls"
    iteration_label = "iteration" if remaining_budget == 1 else "iterations"

    logger.warning(
        "budget_exceeded",
        extra={
            "requested_tool_calls": requested_tool_calls,
            "remaining_budget": remaining_budget,
            "max_iterations": settings.max_iterations,
        },
    )

    return {
        "final_answer": final_answer,
        "outcome": Outcome.BUDGET_EXCEEDED,
        "trajectory": [
            TrajectoryStep(
                tool="max_iterations_guardrail",
                tool_input=state["user_input"],
                output_summary=(
                    f"blocked {requested_tool_calls} {tool_call_label} because only "
                    f"{remaining_budget} {iteration_label} remained"
                ),
            )
        ],
    }


def classify_node(state: DeterministicAgentState) -> dict:
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


def get_fallback_reason(state: DeterministicAgentState) -> FallbackReason | None:
    if state["category"] == Category.UNKNOWN:
        return FallbackReason.UNKNOWN_CATEGORY

    needs_target = state["category"] in (Category.SPECIFIC_CODE, Category.DEPENDENCIES)
    if needs_target and state["target"] is None:
        return FallbackReason.MISSING_TARGET

    return None


def route_by_category(state: DeterministicAgentState) -> str:
    if get_fallback_reason(state) is not None:
        return "generate_response"

    if state["category"] == Category.STRUCTURAL:
        return "run_list_files"
    elif state["category"] == Category.SPECIFIC_CODE:
        return "run_read_file"
    elif state["category"] == Category.DEPENDENCIES:
        return "run_grep"

    raise ValueError(f"Unsupported category for routing: {state['category']}")


def run_list_files_node(state: DeterministicAgentState) -> dict:
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


def run_read_file_node(state: DeterministicAgentState) -> dict:
    """Execute read_file and record the result."""
    relative_path = state["target"]
    if relative_path is None:
        raise ValueError("run_read_file_node requires target to be set")

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


def run_grep_node(state: DeterministicAgentState) -> dict:
    """Execute grep_repo and record the result."""
    term = state["target"]
    if term is None:
        raise ValueError("run_grep_node requires target to be set")

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


def generate_response_node(state: DeterministicAgentState) -> dict:
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
    graph = StateGraph(DeterministicAgentState)

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


def build_react_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the ReAct agent graph.

    Pass a checkpointer to persist conversation state across separate
    invoke() calls sharing the same thread_id (see app.main for how the
    per-turn tool budget and message history are managed on top of that).
    """
    graph = StateGraph(ReActAgentState)

    graph.add_node("agent_decide", agent_decide_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("budget_exceeded", budget_exceeded_node)

    graph.set_entry_point("agent_decide")
    graph.add_conditional_edges(
        "agent_decide",
        route_after_decision,
        {
            "finalize": END,
            "budget_exceeded": "budget_exceeded",
            "execute_tools": "execute_tools",
        },
    )
    graph.add_edge("budget_exceeded", END)
    graph.add_edge("execute_tools", "agent_decide")

    return graph.compile(checkpointer=checkpointer)
