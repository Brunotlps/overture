from pathlib import Path
from typing import Callable

from app.tools import _is_binary_file, list_files, read_file

EmbedFn = Callable[[list[str]], list[list[float]]]


def embed_repo_files(repo_path: str, embed_fn: EmbedFn) -> dict[str, list[float]]:
    """Embed each eligible text file in repo_path using embed_fn.

    Reuses list_files' sensitive-path filtering; binary files are skipped,
    the same guardrail already applied by read_file/grep_repo.
    """
    base = Path(repo_path)
    eligible = []
    contents = []
    for relative_path in list_files(repo_path):
        if _is_binary_file(base / relative_path):
            continue
        eligible.append(relative_path)
        contents.append(read_file(repo_path, relative_path))

    if not eligible:
        return {}

    vectors = embed_fn(contents)
    return dict(zip(eligible, vectors))
