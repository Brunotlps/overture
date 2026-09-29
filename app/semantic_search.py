import hashlib
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.tools import (
    MAX_FILE_LINES,
    _eligible_file,
    _is_binary_file,
    _iter_lines,
    list_files,
)

logger = logging.getLogger(__name__)
EmbedFn = Callable[[list[str]], list[list[float]]]
IndexKey = tuple[str, str, str, str]
SNIPPET_MAX_CHARS = 200
INDEX_POLICY_VERSION = "bounded-head-v2"
MAX_CACHED_INDEXES = 8
# Index admission limits: files past them are counted as skipped, not embedded.
MAX_INDEX_FILES = 1000
MAX_INDEX_FILE_BYTES = 1_000_000
MAX_INDEX_TOTAL_BYTES = 20_000_000
# Embedded text per file: its first MAX_FILE_LINES lines, up to this many chars.
MAX_INDEX_TEXT_CHARS = 20_000
EMBED_BATCH_SIZE = 100


@dataclass(frozen=True)
class SearchResult:
    file_path: str
    score: float
    snippet: str
    # True when only a prefix of the file was embedded.
    partial: bool = False


@dataclass(frozen=True)
class SearchOutcome:
    results: list[SearchResult]
    available: bool
    # Eligible text files left out of the index by its admission limits.
    skipped_files: int = 0


@dataclass(frozen=True)
class RepoIndex:
    vectors: dict[str, list[float]]
    partial_files: frozenset[str]
    skipped_files: int


@dataclass(frozen=True)
class _Snapshot:
    contents: dict[str, str]
    partial_files: frozenset[str]
    skipped_files: int
    fingerprint: str


def _read_index_text(path: Path) -> tuple[str, bool]:
    """Stream the embedded prefix of a file and report whether it is partial."""
    lines: list[str] = []
    chars = 0
    partial = False
    for line_number, (text, omitted) in enumerate(_iter_lines(path), start=1):
        if line_number > MAX_FILE_LINES or chars + len(text) > MAX_INDEX_TEXT_CHARS:
            return "\n".join(lines or [text[:MAX_INDEX_TEXT_CHARS]]), True
        lines.append(text)
        chars += len(text) + 1
        partial = partial or omitted > 0
    return "\n".join(lines), partial


def _snapshot(repo_path: str) -> _Snapshot:
    """Admit text files within the index limits and fingerprint what was admitted.

    Admitted files are fingerprinted by their full bytes (bounded by
    MAX_INDEX_FILE_BYTES), so edits past the embedded prefix still invalidate
    the cache; skipped files contribute their path so the count stays current.
    """
    contents: dict[str, str] = {}
    partial_files = set()
    skipped_files = 0
    admitted_bytes = 0
    digest = hashlib.sha256()
    for relative_path in list_files(repo_path):
        try:
            path = _eligible_file(repo_path, relative_path)
            if _is_binary_file(path):
                continue
            size = path.stat().st_size
            if (
                len(contents) >= MAX_INDEX_FILES
                or size > MAX_INDEX_FILE_BYTES
                or admitted_bytes + size > MAX_INDEX_TOTAL_BYTES
            ):
                skipped_files += 1
                digest.update(b"skipped\0" + relative_path.encode("utf-8") + b"\0")
                continue
            content, partial = _read_index_text(path)
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
        except (ValueError, FileNotFoundError, OSError):
            continue
        admitted_bytes += size
        contents[relative_path] = content
        if partial:
            partial_files.add(relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return _Snapshot(
        contents, frozenset(partial_files), skipped_files, digest.hexdigest()
    )


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
    paths = list(contents)
    vectors: list[list[float]] = []
    dimension = None
    for start in range(0, len(paths), EMBED_BATCH_SIZE):
        batch = [contents[path] for path in paths[start : start + EMBED_BATCH_SIZE]]
        vectors.extend(_validated_vectors(embed_fn(batch), len(batch), dimension))
        dimension = len(vectors[0])
    return dict(zip(paths, vectors, strict=True))


def embed_repo_files(repo_path: str, embed_fn: EmbedFn) -> dict[str, list[float]]:
    """Embed eligible text files using the repository's read policy."""
    return _embed_contents(_snapshot(repo_path).contents, embed_fn)


_index_cache: OrderedDict[IndexKey, RepoIndex] = OrderedDict()
_cache_lock = threading.Lock()
_repo_locks: dict[IndexKey, tuple[threading.Lock, int]] = {}


def get_or_build_index(
    repo_path: str,
    embed_fn: EmbedFn,
    *,
    embedding_identity: str = "default",
    policy_version: str = INDEX_POLICY_VERSION,
) -> RepoIndex:
    """Cache indexes by canonical root, file bytes, embedding config and policy."""
    root = str(Path(repo_path).resolve())
    snapshot = _snapshot(root)
    key = (root, snapshot.fingerprint, embedding_identity, policy_version)
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
            index = RepoIndex(
                _embed_contents(snapshot.contents, embed_fn),
                snapshot.partial_files,
                snapshot.skipped_files,
            )
            logger.info(
                "semantic_index_built",
                extra={
                    "indexed_files": len(index.vectors),
                    "partial_files": len(index.partial_files),
                    "skipped_files": index.skipped_files,
                },
            )
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
    partial_files: frozenset[str] = frozenset(),
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
        snippet, _ = _read_index_text(_eligible_file(repo_path, file_path))
        results.append(
            SearchResult(
                file_path=file_path,
                score=_cosine_similarity(query_vector, vector),
                snippet=snippet[:SNIPPET_MAX_CHARS],
                partial=file_path in partial_files,
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
        results = search(
            query,
            index.vectors,
            embed_fn,
            repo_path,
            top_k=top_k,
            partial_files=index.partial_files,
        )
        return SearchOutcome(results, True, index.skipped_files)
    except Exception as exc:  # noqa: BLE001 - repo tools remain available on failure
        logger.warning(
            "semantic_search_unavailable",
            extra={"repo_path": repo_path, "error_type": type(exc).__name__},
        )
        return SearchOutcome([], False)
