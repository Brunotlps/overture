import os
import fnmatch

from pathlib import Path

IGNORED_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
MAX_FILE_LINES = 300
MAX_GREP_RESULTS_DEFAULT = 20

SENSITIVE_PATTERNS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "*credentials*",
    "*secret*",
    "*token*",
]


def _resolve_within_repo(repo_path: str, relative_path: str) -> Path:
    """Resolve um path relativo garantindo que ele não escape do repo_path."""
    base = Path(repo_path).resolve()
    target = (base / relative_path).resolve()

    if not target.is_relative_to(base):
        raise ValueError(f"Path '{relative_path}' resolves outside repository bounds")

    return target


def list_files(repo_path: str) -> list[str]:
    base = Path(repo_path)
    if not base.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    results = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for filename in files:
            full_path = Path(root) / filename
            relative = full_path.relative_to(base)
            relative_str = str(relative)
            if _is_sensitive_path(relative_str):
                continue
            results.append(relative_str)

    return sorted(results)


def read_file(repo_path: str, relative_path: str) -> str:
    target = _resolve_within_repo(repo_path, relative_path)

    if _is_sensitive_path(relative_path):
        raise ValueError(f"Path '{relative_path}' is blocked because it may contain sensitive data")

    if not target.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")

    lines = target.read_text(errors="replace").splitlines()

    if len(lines) > MAX_FILE_LINES:
        truncated = lines[:MAX_FILE_LINES]
        remaining = len(lines) - MAX_FILE_LINES
        truncated.append(f"... [truncated: {remaining} more lines omitted]")
        return "\n".join(truncated)

    return "\n".join(lines)


def grep_repo(repo_path: str, term: str, max_results: int = MAX_GREP_RESULTS_DEFAULT) -> list[str]:
    base = Path(repo_path)
    if not base.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    matches = []
    for filename in list_files(repo_path):
        full_path = base / filename
        try:
            lines = full_path.read_text(errors="replace").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        for i, line in enumerate(lines, start=1):
            if term in line:
                matches.append(f"{filename}:{i}: {line.strip()}")
                if len(matches) >= max_results:
                    return matches

    return matches

def _is_sensitive_path(relative_path: str) -> bool:
    filename = Path(relative_path).name
    return any(fnmatch.fnmatch(filename, pattern) for pattern in SENSITIVE_PATTERNS)
