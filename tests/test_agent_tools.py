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
