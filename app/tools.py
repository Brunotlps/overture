import fnmatch
import os
from collections.abc import Iterator
from pathlib import Path

IGNORED_DIRS = {".git", ".claude", "__pycache__", "node_modules", ".venv"}
MAX_FILE_LINES = 300
# Per-line cap shared by read_file and grep_repo, so anything grep matches is
# also visible to read_file; text past it on a single line is never searched.
MAX_LINE_CHARS = 2000
MAX_READ_CHARS = 20_000
LINE_SKIP_CHUNK_CHARS = 64 * 1024
MAX_LIST_FILES = 200
MAX_LIST_CHARS = 20_000
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


def list_files_page(
    repo_path: str, offset: int = 0, limit: int = MAX_LIST_FILES
) -> str:
    """Return one page of the sorted listing, ending with the offset to continue."""
    if offset < 0:
        raise ValueError("offset must be 0 or greater")
    if not 1 <= limit <= MAX_LIST_FILES:
        raise ValueError(f"limit must be between 1 and {MAX_LIST_FILES}")

    files = list_files(repo_path)
    if offset and offset >= len(files):
        raise ValueError(
            f"offset {offset} is past the end of the listing ({len(files)} files)"
        )

    page: list[str] = []
    page_chars = 0
    for path in files[offset : offset + limit]:
        if page and page_chars + len(path) > MAX_LIST_CHARS:
            break
        page.append(path)
        page_chars += len(path) + 1

    end = offset + len(page)
    if end < len(files):
        page.append(
            f"... [showing files {offset + 1}-{end} of {len(files)}; "
            f"call list_files with offset={end} to continue]"
        )
    return "\n".join(page)


def _iter_lines(path: Path) -> Iterator[tuple[str, int]]:
    """Stream (text, omitted_chars) per line, holding at most MAX_LINE_CHARS of a line.

    Universal newlines (\\n, \\r\\n, \\r) define line numbers for both read_file
    and grep_repo, so a line number reported by one addresses the other.
    """
    with path.open(errors="replace") as handle:
        while line := handle.readline(MAX_LINE_CHARS + 1):
            text = line.removesuffix("\n")
            omitted = 0
            if len(text) > MAX_LINE_CHARS:
                omitted = len(text) - MAX_LINE_CHARS
                text = text[:MAX_LINE_CHARS]
                while not line.endswith("\n") and (
                    line := handle.readline(LINE_SKIP_CHUNK_CHARS)
                ):
                    omitted += len(line.removesuffix("\n"))
            yield text, omitted


def read_file(
    repo_path: str,
    relative_path: str,
    start_line: int = 1,
    max_lines: int = MAX_FILE_LINES,
) -> str:
    """Read a numbered line range, stopping once the range or output cap is filled."""
    if start_line < 1:
        raise ValueError("start_line must be 1 or greater")
    if not 1 <= max_lines <= MAX_FILE_LINES:
        raise ValueError(f"max_lines must be between 1 and {MAX_FILE_LINES}")

    target = _eligible_file(repo_path, relative_path)

    if _is_binary_file(target):
        raise ValueError(f"File '{relative_path}' is binary and cannot be read as text")

    output: list[str] = []
    output_chars = 0
    total_lines = 0
    for line_number, (text, omitted) in enumerate(_iter_lines(target), start=1):
        total_lines = line_number
        if line_number < start_line:
            continue
        rendered = f"{line_number}: {text}"
        if omitted:
            rendered += f" ... [line truncated: {omitted} more characters]"
        if len(output) == max_lines or output_chars + len(rendered) > MAX_READ_CHARS:
            output.append(
                f"... [more lines follow; call read_file with start_line={line_number}"
                " to continue]"
            )
            break
        output.append(rendered)
        output_chars += len(rendered) + 1

    if start_line > max(total_lines, 1):
        raise ValueError(
            f"start_line {start_line} is past the end of '{relative_path}' "
            f"({total_lines} lines)"
        )
    return "\n".join(output)


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
            for i, (line, _) in enumerate(_iter_lines(full_path), start=1):
                if term in line:
                    snippet = line.strip()
                    if len(snippet) > MAX_GREP_LINE_CHARS:
                        snippet = snippet[:MAX_GREP_LINE_CHARS] + "... [truncated]"
                    matches.append(f"{filename}:{i}: {snippet}")
                    if len(matches) >= max_results:
                        return matches
        except OSError:
            continue

    return matches

def _is_sensitive_path(relative_path: str) -> bool:
    return any(
        fnmatch.fnmatch(part.casefold(), pattern)
        for part in Path(relative_path).parts
        for pattern in SENSITIVE_PATTERNS
    )
