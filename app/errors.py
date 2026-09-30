"""Stable, content-free error descriptions for model and client boundaries."""


def tool_error_code(exc: Exception) -> str:
    """Classify expected repository tool failures without inspecting their text."""
    if isinstance(exc, FileNotFoundError):
        return "file_not_found"
    if isinstance(exc, ValueError):
        return "invalid_input"
    if isinstance(exc, OSError):
        return "filesystem_error"
    return "tool_error"


def tool_error_message(code: str) -> str:
    """Give the agent enough direction to recover without provider details."""
    guidance = {
        "file_not_found": "Check the path or list files and try again.",
        "invalid_input": "Check the arguments and try again.",
        "filesystem_error": "The repository could not be read; try another tool.",
        "unknown_tool": "Choose an available repository tool.",
    }.get(code, "Try another repository tool.")
    return f"Tool error: {code}. {guidance}"
