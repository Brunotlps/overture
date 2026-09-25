import fnmatch
import os

from pathlib import Path

IGNORED_DIRS = {".git", ".claude", "__pycache__", "node_modules", ".venv"}
MAX_FILE_LINES = 300
MAX_GREP_RESULTS_DEFAULT = 20
MAX_GREP_LINE_CHARS = 200
BINARY_SNIFF_BYTES = 1024

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


def _eligible_file(repo_path: str, relative_path: str) -> Path:
    """Validate the same file policy for listing, reading, and searching."""
    base = Path(repo_path).resolve()
    requested = Path(relative_path)
    for part in requested.parts:
        if part.casefold() in IGNORED_DIRS:
            raise ValueError(f"Path '{relative_path}' is inside an ignored directory")
    if _is_sensitive_path(relative_path):
        raise ValueError(f"Path '{relative_path}' is blocked because it may contain sensitive data")

    current = base
    for part in (() if requested.is_absolute() else requested.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Path '{relative_path}' contains a symlink")

    target = _resolve_within_repo(repo_path, relative_path)
    if _is_sensitive_path(str(target.relative_to(base))):
        raise ValueError(f"Path '{relative_path}' is blocked because it may contain sensitive data")
    if not target.exists():
        raise FileNotFoundError(f"File not found: {relative_path}")
    if not target.is_file():
        raise ValueError(f"Path '{relative_path}' is not a file")
    return target


def _is_binary_file(path: Path) -> bool:
    """Detecta arquivos binários pela presença de byte nulo no início do arquivo."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return True


def list_files(repo_path: str) -> list[str]:
    base = Path(repo_path)
    if not base.exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    results = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [
            d
            for d in dirs
            if d.casefold() not in IGNORED_DIRS
            and not _is_sensitive_path(d)
            and not (Path(root) / d).is_symlink()
        ]
        for filename in files:
            full_path = Path(root) / filename
            relative = full_path.relative_to(base)
            relative_str = str(relative)
            try:
                _eligible_file(repo_path, relative_str)
            except (ValueError, FileNotFoundError, OSError):
                continue
            results.append(relative_str)

    return sorted(results)


def read_file(repo_path: str, relative_path: str) -> str:
    target = _eligible_file(repo_path, relative_path)

    if _is_binary_file(target):
        raise ValueError(f"File '{relative_path}' is binary and cannot be read as text")

    lines = target.read_text(errors="replace").splitlines()

    if len(lines) > MAX_FILE_LINES:
        truncated = lines[:MAX_FILE_LINES]
        remaining = len(lines) - MAX_FILE_LINES
        truncated.append(f"... [truncated: {remaining} more lines omitted]")
        return "\n".join(truncated)

    return "\n".join(lines)


def grep_repo(repo_path: str, term: str, max_results: int = MAX_GREP_RESULTS_DEFAULT) -> list[str]:
    if not Path(repo_path).exists():
        raise FileNotFoundError(f"Repository path not found: {repo_path}")

    matches = []
    for filename in list_files(repo_path):
        try:
            full_path = _eligible_file(repo_path, filename)
        except (ValueError, FileNotFoundError, OSError):
            continue
        if _is_binary_file(full_path):
            continue

        try:
            lines = full_path.read_text(errors="replace").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        for i, line in enumerate(lines, start=1):
            if term in line:
                snippet = line.strip()
                if len(snippet) > MAX_GREP_LINE_CHARS:
                    snippet = snippet[:MAX_GREP_LINE_CHARS] + "... [truncated]"
                matches.append(f"{filename}:{i}: {snippet}")
                if len(matches) >= max_results:
                    return matches

    return matches

def _is_sensitive_path(relative_path: str) -> bool:
    return any(
        fnmatch.fnmatch(part.casefold(), pattern)
        for part in Path(relative_path).parts
        for pattern in SENSITIVE_PATTERNS
    )
