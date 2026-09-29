from app import agent_tools
from app.agent_tools import get_llm_tools, get_tool_registry
from app.config import settings
from app.semantic_search import SearchOutcome


def test_semantic_search_tool_absent_when_feature_flag_off(monkeypatch):
    monkeypatch.setattr(settings, "semantic_search_enabled", False)

    tool_names = {t.name for t in get_llm_tools()}

    assert "semantic_search" not in tool_names
    assert "semantic_search" not in get_tool_registry()


def test_semantic_search_tool_present_when_feature_flag_on(monkeypatch):
    monkeypatch.setattr(settings, "semantic_search_enabled", True)

    tool_names = {t.name for t in get_llm_tools()}

    assert "semantic_search" in tool_names
    assert "semantic_search" in get_tool_registry()


def test_embedding_model_and_endpoint_are_explicit(monkeypatch):
    arguments = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            arguments.update(kwargs)

        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

    monkeypatch.setattr(agent_tools, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(settings, "embedding_model", "test-model")
    monkeypatch.setattr(settings, "embedding_base_url", "https://embeddings.example/v1")
    monkeypatch.setattr(settings, "embedding_api_key", "embedding-key")

    assert agent_tools._embed_fn(["query"]) == [[1.0]]
    assert arguments == {
        "model": "test-model",
        "base_url": "https://embeddings.example/v1",
        "api_key": "embedding-key",
    }


def test_unavailable_search_is_distinct_from_no_results():
    assert agent_tools._format_results(SearchOutcome([], False)).startswith(
        "Semantic search unavailable"
    )
    assert agent_tools._format_results(SearchOutcome([], True)) == "No results."


def test_tool_cache_identity_changes_with_embedding_configuration(monkeypatch):
    identities = []

    def fake_search(query, repo_path, embed_fn, *, embedding_identity):
        identities.append(embedding_identity)
        return SearchOutcome([], True)

    monkeypatch.setattr(agent_tools, "run_semantic_search", fake_search)
    monkeypatch.setattr(settings, "embedding_model", "model-a")
    assert agent_tools.semantic_search_tool.func("query", "/repo") == "No results."
    monkeypatch.setattr(settings, "embedding_model", "model-b")
    assert agent_tools.semantic_search_tool.func("query", "/repo") == "No results."
    assert identities[0] != identities[1]


def test_read_file_tool_exposes_line_range_to_the_model(tmp_path):
    (tmp_path / "long.py").write_text("".join(f"value_{i} = {i}\n" for i in range(1, 401)))
    read_file_tool = get_tool_registry()["read_file"]

    schema = read_file_tool.tool_call_schema.model_json_schema()["properties"]
    content = read_file_tool.invoke(
        {
            "relative_path": "long.py",
            "start_line": 350,
            "max_lines": 2,
            "repo_path": str(tmp_path),
        }
    )

    assert set(schema) == {"relative_path", "start_line", "max_lines"}
    assert content.splitlines() == [
        "350: value_350 = 350",
        "351: value_351 = 351",
        "... [more lines follow; call read_file with start_line=352 to continue]",
    ]
