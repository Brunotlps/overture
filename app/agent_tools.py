from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg, tool

from app.tools import grep_repo, list_files, read_file


@tool("list_files")
def list_files_tool(repo_path: Annotated[str, InjectedToolArg]) -> str:
    """List non-sensitive files in the configured repository."""
    return "\n".join(list_files(repo_path))


@tool("read_file")
def read_file_tool(
    relative_path: str, repo_path: Annotated[str, InjectedToolArg]
) -> str:
    """Read a non-sensitive file from the configured repository."""
    return read_file(repo_path, relative_path)


@tool("grep_repo")
def grep_repo_tool(term: str, repo_path: Annotated[str, InjectedToolArg]) -> str:
    """Search for a term across non-sensitive files in the configured repository."""
    return "\n".join(grep_repo(repo_path, term))


def get_llm_tools() -> list[BaseTool]:
    """Return the tool definitions exposed to the LLM."""
    return [list_files_tool, read_file_tool, grep_repo_tool]


def get_tool_registry() -> dict[str, BaseTool]:
    """Return the executable tool registry used by the agent tool executor."""
    return {
        "list_files": list_files_tool,
        "read_file": read_file_tool,
        "grep_repo": grep_repo_tool,
    }
