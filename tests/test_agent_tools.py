from app.agent_tools import get_llm_tools, get_tool_registry
from app.config import settings


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
