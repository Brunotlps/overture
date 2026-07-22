from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.semantic_search import SearchResult
from app.semantic_search import semantic_search as run_semantic_search
from app.tools import grep_repo, list_files, read_file


def _embed_fn(texts: list[str]) -> list[list[float]]:
    embeddings = OpenAIEmbeddings(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key
    )
    return embeddings.embed_documents(texts)


def _format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results."
    return "\n".join(
        f"{r.file_path} (score={r.score:.3f}): {r.snippet}" for r in results
    )


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


@tool("semantic_search")
def semantic_search_tool(
    query: str, repo_path: Annotated[str, InjectedToolArg]
) -> str:
    """Find files by meaning when grep_repo misses (no lexical overlap)."""
    return _format_results(run_semantic_search(query, repo_path, _embed_fn))


def get_llm_tools() -> list[BaseTool]:
    """Return the tool definitions exposed to the LLM."""
    tools = [list_files_tool, read_file_tool, grep_repo_tool]
    if settings.semantic_search_enabled:
        tools.append(semantic_search_tool)
    return tools


def get_tool_registry() -> dict[str, BaseTool]:
    """Return the executable tool registry used by the agent tool executor."""
    registry = {
        "list_files": list_files_tool,
        "read_file": read_file_tool,
        "grep_repo": grep_repo_tool,
    }
    if settings.semantic_search_enabled:
        registry["semantic_search"] = semantic_search_tool
    return registry
