from app.semantic_search import (
    SearchResult,
    embed_repo_files,
    get_or_build_index,
    search,
    semantic_search,
)


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

    assert results == []
