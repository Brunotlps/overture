import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.tools import _is_binary_file, list_files, read_file

EmbedFn = Callable[[list[str]], list[list[float]]]

SNIPPET_MAX_CHARS = 200


@dataclass(frozen=True)
class SearchResult:
    file_path: str
    score: float
    snippet: str


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


_index_cache: dict[str, dict[str, list[float]]] = {}
_cache_lock = threading.Lock()
_repo_locks: dict[str, threading.Lock] = {}


def _get_repo_lock(repo_path: str) -> threading.Lock:
    with _cache_lock:
        if repo_path not in _repo_locks:
            _repo_locks[repo_path] = threading.Lock()
        return _repo_locks[repo_path]


def get_or_build_index(repo_path: str, embed_fn: EmbedFn) -> dict[str, list[float]]:
    """Build repo_path's embedding index on first use, cached for the process.

    A per-repo_path lock prevents two concurrent first-uses of the same repo
    from triggering duplicate (costly) embedding calls.
    """
    if repo_path in _index_cache:
        return _index_cache[repo_path]

    with _get_repo_lock(repo_path):
        if repo_path not in _index_cache:
            _index_cache[repo_path] = embed_repo_files(repo_path, embed_fn)

    return _index_cache[repo_path]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(
    query: str,
    index: dict[str, list[float]],
    embed_fn: EmbedFn,
    repo_path: str,
    top_k: int = 3,
) -> list[SearchResult]:
    if not index:
        return []

    query_vector = embed_fn([query])[0]
    ranked = sorted(
        index.items(),
        key=lambda item: _cosine_similarity(query_vector, item[1]),
        reverse=True,
    )

    results = []
    for file_path, vector in ranked[:top_k]:
        snippet = read_file(repo_path, file_path)[:SNIPPET_MAX_CHARS]
        results.append(
            SearchResult(
                file_path=file_path,
                score=_cosine_similarity(query_vector, vector),
                snippet=snippet,
            )
        )
    return results
