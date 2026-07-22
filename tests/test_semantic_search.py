from app.semantic_search import embed_repo_files


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
