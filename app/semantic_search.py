import hashlib
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.tools import MAX_FILE_LINES, _eligible_file, _is_binary_file, list_files

logger = logging.getLogger(__name__)
EmbedFn = Callable[[list[str]], list[list[float]]]
IndexKey = tuple[str, str, str, str]
SNIPPET_MAX_CHARS = 200
INDEX_POLICY_VERSION = "whole-file-v1"
MAX_CACHED_INDEXES = 8


@dataclass(frozen=True)
class SearchResult:
    file_path: str
    score: float
    snippet: str


@dataclass(frozen=True)
class SearchOutcome:
    results: list[SearchResult]
    available: bool


def _read_index_text(repo_path: str, relative_path: str) -> str:
    """Return the text embedded for a file: its first MAX_FILE_LINES lines."""
    target = _eligible_file(repo_path, relative_path)
    if _is_binary_file(target):
        raise ValueError(f"File '{relative_path}' is binary and cannot be read as text")

    lines = target.read_text(errors="replace").splitlines()
    if len(lines) > MAX_FILE_LINES:
        remaining = len(lines) - MAX_FILE_LINES
        return "\n".join(
            [*lines[:MAX_FILE_LINES], f"... [truncated: {remaining} more lines omitted]"]
        )
    return "\n".join(lines)


def _snapshot(repo_path: str) -> tuple[dict[str, str], str]:
    """Read eligible text and fingerprint full file bytes, including truncated tails."""
    contents = {}
    digest = hashlib.sha256()
    for relative_path in list_files(repo_path):
        try:
            path = _eligible_file(repo_path, relative_path)
            content = _read_index_text(repo_path, relative_path)
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
        except (ValueError, FileNotFoundError, OSError):
            continue
        contents[relative_path] = content
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return contents, digest.hexdigest()


def _validated_vectors(
    vectors: object, count: int, dimension: int | None = None
) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedding provider returned the wrong number of vectors")
    result = []
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Embedding provider returned an empty or invalid vector")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                "Embedding provider returned incompatible vector dimensions or values"
            )
        result.append(vector)
    return result


def _embed_contents(
    contents: dict[str, str], embed_fn: EmbedFn
) -> dict[str, list[float]]:
    if not contents:
        return {}
    vectors = _validated_vectors(embed_fn(list(contents.values())), len(contents))
    return dict(zip(contents, vectors, strict=True))


def embed_repo_files(repo_path: str, embed_fn: EmbedFn) -> dict[str, list[float]]:
    """Embed eligible text files using the repository's read policy."""
    contents, _ = _snapshot(repo_path)
    return _embed_contents(contents, embed_fn)


_index_cache: OrderedDict[IndexKey, dict[str, list[float]]] = OrderedDict()
_cache_lock = threading.Lock()
_repo_locks: dict[IndexKey, tuple[threading.Lock, int]] = {}


def get_or_build_index(
    repo_path: str,
    embed_fn: EmbedFn,
    *,
    embedding_identity: str = "default",
    policy_version: str = INDEX_POLICY_VERSION,
) -> dict[str, list[float]]:
    """Cache indexes by canonical root, file bytes, embedding config and policy."""
    root = str(Path(repo_path).resolve())
    contents, fingerprint = _snapshot(root)
    key = (root, fingerprint, embedding_identity, policy_version)
    with _cache_lock:
        cached = _index_cache.get(key)
        if cached is not None:
            _index_cache.move_to_end(key)
            return cached
        lock, users = _repo_locks.get(key, (threading.Lock(), 0))
        _repo_locks[key] = (lock, users + 1)

    try:
        with lock:
            with _cache_lock:
                cached = _index_cache.get(key)
                if cached is not None:
                    _index_cache.move_to_end(key)
                    return cached
            index = _embed_contents(contents, embed_fn)
            with _cache_lock:
                _index_cache[key] = index
                _index_cache.move_to_end(key)
                if len(_index_cache) > MAX_CACHED_INDEXES:
                    _index_cache.popitem(last=False)
            return index
    finally:
        with _cache_lock:
            current_lock, users = _repo_locks[key]
            if users == 1:
                del _repo_locks[key]
            else:
                _repo_locks[key] = (current_lock, users - 1)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding vectors have incompatible dimensions")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
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

    dimension = len(next(iter(index.values())))
    query_vector = _validated_vectors(embed_fn([query]), 1, dimension)[0]
    ranked = sorted(
        index.items(),
        key=lambda item: _cosine_similarity(query_vector, item[1]),
        reverse=True,
    )
    results = []
    for file_path, vector in ranked[:top_k]:
        snippet = _read_index_text(repo_path, file_path)[:SNIPPET_MAX_CHARS]
        results.append(
            SearchResult(
                file_path=file_path,
                score=_cosine_similarity(query_vector, vector),
                snippet=snippet,
            )
        )
    return results


def semantic_search(
    query: str,
    repo_path: str,
    embed_fn: EmbedFn,
    top_k: int = 3,
    *,
    embedding_identity: str = "default",
) -> SearchOutcome:
    """Return results or an observable unavailable outcome on provider failure."""
    try:
        index = get_or_build_index(
            repo_path, embed_fn, embedding_identity=embedding_identity
        )
        return SearchOutcome(
            search(query, index, embed_fn, repo_path, top_k=top_k), True
        )
    except Exception as exc:  # noqa: BLE001 - repo tools remain available on failure
        logger.warning(
            "semantic_search_unavailable",
            extra={"repo_path": repo_path, "error_type": type(exc).__name__},
        )
        return SearchOutcome([], False)
