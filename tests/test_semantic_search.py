import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import semantic_search as module
from app.semantic_search import (
    SearchOutcome,
    SearchResult,
    embed_repo_files,
    get_or_build_index,
    search,
    semantic_search,
)
from app.tools import MAX_LINE_CHARS


def test_embeds_each_eligible_file_in_repo(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass")
    (tmp_path / "b.py").write_text("def b(): pass")

    calls = []

    def fake_embed_fn(texts):
        calls.append(texts)
        return [[float(len(text)), 0.0] for text in texts]

    index = embed_repo_files(str(tmp_path), fake_embed_fn)

    assert set(index.keys()) == {"a.py", "b.py"}
    assert index["a.py"] == [float(len("def a(): pass")), 0.0]
    assert index["b.py"] == [float(len("def b(): pass")), 0.0]
    assert len(calls) == 1  # embeds all files in a single batch call


def test_long_file_embeds_an_unnumbered_prefix_and_is_marked_partial(tmp_path):
    (tmp_path / "long.py").write_text("".join(f"row {i}\n" for i in range(305)))
    (tmp_path / "short.py").write_text("row\n")
    texts = []

    index = get_or_build_index(
        str(tmp_path), lambda batch: texts.extend(batch) or [[1.0] for _ in batch]
    )

    assert texts == ["\n".join(f"row {i}" for i in range(300)), "row"]
    assert index.partial_files == {"long.py"}
    assert index.skipped_files == 0


def test_index_text_is_capped_by_long_lines_and_characters(tmp_path, monkeypatch):
    (tmp_path / "minified.js").write_text("x" * (MAX_LINE_CHARS + 10))
    (tmp_path / "wide.py").write_text("0123456789\n" * 5)
    texts = []

    def embed(batch):
        texts.append(batch)
        return [[1.0] for _ in batch]

    first = get_or_build_index(str(tmp_path), embed)
    monkeypatch.setattr(module, "MAX_INDEX_TEXT_CHARS", 25)
    second = get_or_build_index(str(tmp_path), embed, policy_version="capped")
    monkeypatch.setattr(module, "MAX_INDEX_TEXT_CHARS", 5)
    third = get_or_build_index(str(tmp_path), embed, policy_version="tiny")

    assert texts[0] == ["x" * MAX_LINE_CHARS, "0123456789\n" * 4 + "0123456789"]
    assert first.partial_files == {"minified.js"}
    assert texts[1] == ["x" * 25, "0123456789\n0123456789"]
    assert second.partial_files == {"minified.js", "wide.py"}
    assert texts[2] == ["xxxxx", "01234"]
    assert third.partial_files == {"minified.js", "wide.py"}


def test_admission_limits_skip_files_and_report_partial_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_INDEX_FILES", 2)
    monkeypatch.setattr(module, "MAX_INDEX_FILE_BYTES", 10)
    for name, content in [
        ("a.py", "a"),
        ("b.py", "b" * 11),
        ("c.py", "c"),
        ("d.py", "d"),
    ]:
        (tmp_path / name).write_text(content)
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00" * 10)
    embed = lambda batch: [[1.0, 0.0] for _ in batch]

    index = get_or_build_index(str(tmp_path), embed)
    outcome = semantic_search("query", str(tmp_path), embed)

    assert set(index.vectors) == {"a.py", "c.py"}
    assert index.skipped_files == 2
    assert outcome.skipped_files == 2
    assert all(not result.partial for result in outcome.results)


def test_total_bytes_limit_skips_files_that_do_not_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_INDEX_TOTAL_BYTES", 10)
    (tmp_path / "a.py").write_text("a" * 6)
    (tmp_path / "b.py").write_text("b" * 6)
    (tmp_path / "c.py").write_text("c" * 4)

    index = get_or_build_index(str(tmp_path), lambda batch: [[1.0] for _ in batch])

    assert set(index.vectors) == {"a.py", "c.py"}
    assert index.skipped_files == 1


def test_many_files_stay_within_limits_without_reading_skipped_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "MAX_INDEX_FILES", 50)
    for i in range(300):
        (tmp_path / f"m{i:03}.py").write_text(f"value = {i}\n")
    read_paths = []
    real_read = module._read_index_text
    monkeypatch.setattr(
        module,
        "_read_index_text",
        lambda path: read_paths.append(path.name) or real_read(path),
    )

    index = get_or_build_index(str(tmp_path), lambda batch: [[1.0] for _ in batch])

    assert len(index.vectors) == len(read_paths) == 50
    assert index.skipped_files == 250


def test_skipped_files_changes_invalidate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MAX_INDEX_FILES", 1)
    (tmp_path / "a.py").write_text("a")
    embed = lambda batch: [[1.0] for _ in batch]
    first = get_or_build_index(str(tmp_path), embed)

    (tmp_path / "b.py").write_text("b")
    second = get_or_build_index(str(tmp_path), embed)

    assert second is not first
    assert (first.skipped_files, second.skipped_files) == (0, 1)


def test_embeds_in_bounded_batches_with_consistent_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "EMBED_BATCH_SIZE", 2)
    for name in ("a.py", "b.py", "c.py", "d.py", "e.py"):
        (tmp_path / name).write_text(name)
    batches = []

    def embed(batch):
        batches.append(batch)
        return [[1.0, 0.0] for _ in batch]

    index = embed_repo_files(str(tmp_path), embed)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert set(index) == {"a.py", "b.py", "c.py", "d.py", "e.py"}
    dimensions = iter([[[1.0]] * 2, [[1.0, 0.0]] * 2, [[1.0]]])
    with pytest.raises(ValueError, match="incompatible vector dimensions"):
        embed_repo_files(str(tmp_path), lambda batch: next(dimensions))


def test_index_skips_symlinks_without_dropping_safe_files(tmp_path):
    (tmp_path / "safe.py").write_text("public")
    (tmp_path / "SECRET.txt").write_text("private")
    (tmp_path / "alias.py").symlink_to("SECRET.txt")

    index = embed_repo_files(
        str(tmp_path), lambda texts: [[float(len(text))] for text in texts]
    )

    assert index == {"safe.py": [6.0]}


def test_returns_top_k_results_ranked_by_similarity(tmp_path):
    (tmp_path / "close.py").write_text("money handling code")
    (tmp_path / "medium.py").write_text("somewhat related code")
    (tmp_path / "far.py").write_text("totally unrelated code")

    index = {
        "close.py": [1.0, 0.0],
        "medium.py": [0.7, 0.7],
        "far.py": [0.0, 1.0],
    }

    def fake_embed_fn(texts):
        return [[1.0, 0.0] for _ in texts]  # query vector == "close.py"

    results = search(
        "how is money handled?", index, fake_embed_fn, str(tmp_path), top_k=2
    )

    assert [r.file_path for r in results] == ["close.py", "medium.py"]
    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].score > results[1].score


def test_result_snippet_is_truncated_file_content(tmp_path):
    long_content = "x" * 500
    (tmp_path / "big.py").write_text(long_content)

    index = {"big.py": [1.0, 0.0]}

    def fake_embed_fn(texts):
        return [[1.0, 0.0] for _ in texts]

    [result] = search("query", index, fake_embed_fn, str(tmp_path), top_k=1)

    assert result.snippet == long_content[:200]
    assert len(result.snippet) == 200


def test_index_is_built_only_once_per_repo_path_across_multiple_calls(tmp_path):
    (tmp_path / "a.py").write_text("content")

    calls = []

    def fake_embed_fn(texts):
        calls.append(texts)
        return [[1.0] for _ in texts]

    get_or_build_index(str(tmp_path), fake_embed_fn)
    get_or_build_index(str(tmp_path), fake_embed_fn)

    assert len(calls) == 1


def test_search_degrades_gracefully_when_embedding_fails(tmp_path):
    (tmp_path / "a.py").write_text("content")

    def failing_embed_fn(texts):
        raise RuntimeError("embedding provider unavailable")

    results = semantic_search("some query", str(tmp_path), failing_embed_fn)

    assert results == SearchOutcome([], False)


def test_empty_repo_is_available_without_calling_provider(tmp_path):
    outcome = semantic_search("query", str(tmp_path), lambda _: pytest.fail("called"))
    assert outcome == SearchOutcome([], True)


def test_added_edited_and_removed_files_invalidate_cache(tmp_path):
    (tmp_path / "a.py").write_text("original")
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[float(len(text))] for text in texts]

    first = get_or_build_index(str(tmp_path), embed)
    (tmp_path / "b.py").write_text("new")
    second = get_or_build_index(str(tmp_path), embed)
    assert set(second.vectors) == {"a.py", "b.py"}
    assert second is not first

    (tmp_path / "a.py").write_text("changed")
    third = get_or_build_index(str(tmp_path), embed)
    assert third.vectors["a.py"] == [7.0]
    assert third is not second

    (tmp_path / "b.py").unlink()
    fourth = get_or_build_index(str(tmp_path), embed)
    assert set(fourth.vectors) == {"a.py"}
    assert fourth is not third
    assert len(calls) == 4


def test_changes_after_read_limit_still_invalidate_cache(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("line\n" * 300 + "tail one\n")
    embed = lambda texts: [[1.0] for _ in texts]
    first = get_or_build_index(str(tmp_path), embed)
    path.write_text("line\n" * 300 + "tail two\n")
    assert get_or_build_index(str(tmp_path), embed) is not first


def test_canonical_root_model_and_policy_are_cache_keys(tmp_path):
    (tmp_path / "a.py").write_text("content")
    embed = lambda texts: [[1.0] for _ in texts]
    original = get_or_build_index(str(tmp_path), embed, embedding_identity="model-a")
    assert (
        get_or_build_index(str(tmp_path / "."), embed, embedding_identity="model-a")
        is original
    )
    assert (
        get_or_build_index(str(tmp_path), embed, embedding_identity="model-b")
        is not original
    )
    assert (
        get_or_build_index(
            str(tmp_path), embed, embedding_identity="model-a", policy_version="v2"
        )
        is not original
    )


@pytest.mark.parametrize(
    "vectors",
    [[], [[1.0]], [[1.0], [1.0, 2.0]], [[1.0], [math.nan]], [[1.0], [math.inf]]],
)
def test_rejects_invalid_file_embeddings(tmp_path, vectors):
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    with pytest.raises(ValueError):
        embed_repo_files(str(tmp_path), lambda _: vectors)


@pytest.mark.parametrize("vector", [[], [1.0], [math.nan, 0.0]])
def test_rejects_invalid_query_embedding(tmp_path, vector):
    (tmp_path / "a.py").write_text("a")
    with pytest.raises(ValueError):
        search("query", {"a.py": [1.0, 0.0]}, lambda _: [vector], str(tmp_path))


def test_cache_and_locks_are_bounded(tmp_path):
    path = tmp_path / "a.py"
    embed = lambda texts: [[1.0] for _ in texts]
    for revision in range(module.MAX_CACHED_INDEXES + 3):
        path.write_text(str(revision))
        get_or_build_index(str(tmp_path), embed)
    assert len(module._index_cache) <= module.MAX_CACHED_INDEXES
    assert not module._repo_locks


def test_concurrent_first_use_embeds_only_once(tmp_path):
    (tmp_path / "a.py").write_text("a")
    gate = threading.Barrier(2)
    calls = []

    def embed(texts):
        calls.append(texts)
        time.sleep(0.05)
        return [[1.0] for _ in texts]

    def build():
        gate.wait()
        return get_or_build_index(str(tmp_path), embed)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(build)
        second = pool.submit(build)
        assert first.result() is second.result()
    assert len(calls) == 1
    assert not module._repo_locks
