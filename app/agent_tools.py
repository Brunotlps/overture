import hashlib
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg, tool
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.semantic_search import SearchOutcome
from app.semantic_search import semantic_search as run_semantic_search
from app.tools import (
    MAX_FILE_LINES,
    MAX_LIST_FILES,
    grep_repo,
    list_files_page,
    read_file,
)
from app.usage import ModelInputTooLarge, remaining_provider_timeout


def _embed_fn(texts: list[str]) -> list[list[float]]:
    if sum(len(text) for text in texts) > settings.model_max_input_chars:
        raise ModelInputTooLarge("Embedding input exceeds the configured context budget")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url or settings.llm_base_url,
        api_key=settings.embedding_api_key or settings.llm_api_key,
        timeout=remaining_provider_timeout(settings.provider_timeout_seconds),
        max_retries=settings.provider_max_retries,
    )
    return embeddings.embed_documents(texts)


def _format_results(outcome: SearchOutcome) -> str:
    if not outcome.available:
        return "Semantic search unavailable. Use grep_repo or read_file."
    lines = [
        f"{r.file_path} (score={r.score:.3f})"
        + (" [indexed from its first lines only]" if r.partial else "")
        + f": {r.snippet}"
        for r in outcome.results
    ] or ["No results."]
    if outcome.skipped_files:
        lines.append(
            f"Note: {outcome.skipped_files} files were left out of the semantic index "
            "by its size limits; use grep_repo or list_files to cover them."
        )
    return "\n".join(lines)


@tool("list_files")
def list_files_tool(
    repo_path: Annotated[str, InjectedToolArg],
    offset: int = 0,
    limit: int = MAX_LIST_FILES,
) -> str:
    """List non-sensitive files in the configured repository, sorted by path.

    Returns up to limit paths (at most 200) starting at offset (0-based). When
    more files follow, the output ends with the offset to pass to continue.
    """
    return list_files_page(repo_path, offset, limit)


@tool("read_file")
def read_file_tool(
    relative_path: str,
    repo_path: Annotated[str, InjectedToolArg],
    start_line: int = 1,
    max_lines: int = MAX_FILE_LINES,
) -> str:
    """Read numbered lines of a non-sensitive file from the configured repository.

    Returns up to max_lines lines (at most 300) starting at start_line (1-based),
    each prefixed with its line number. When more lines follow, the output ends
    with the start_line to pass to continue reading.
    """
    return read_file(repo_path, relative_path, start_line, max_lines)


@tool("grep_repo")
def grep_repo_tool(term: str, repo_path: Annotated[str, InjectedToolArg]) -> str:
    """Search for a term across non-sensitive files in the configured repository."""
    return "\n".join(grep_repo(repo_path, term))


@tool("semantic_search")
def semantic_search_tool(query: str, repo_path: Annotated[str, InjectedToolArg]) -> str:
    """Find files by meaning when grep_repo misses (no lexical overlap)."""
    endpoint = settings.embedding_base_url or settings.llm_base_url
    credential = settings.embedding_api_key or settings.llm_api_key
    identity = hashlib.sha256(
        f"{settings.embedding_model}\0{endpoint}\0{credential}".encode()
    ).hexdigest()
    return _format_results(
        run_semantic_search(query, repo_path, _embed_fn, embedding_identity=identity)
    )


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
