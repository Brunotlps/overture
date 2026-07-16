import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Category, ClassificationResult


class FakeStructuredLLM:
    def __init__(self, result, parent):
        self._result = result
        self._parent = parent

    def invoke(self, prompt):
        self._parent.structured_invocations += 1
        self._parent.last_structured_prompt = prompt
        return self._result


class FakeLLM:
    def __init__(self, structured_result=None, text_response="fake answer"):
        self._structured_result = structured_result or ClassificationResult(
            category=Category.UNKNOWN,
            reasoning="fake classification",
        )
        self._text_response = text_response
        self.structured_invocations = 0
        self.text_invocations = 0
        self.last_structured_prompt = None
        self.last_text_prompt = None

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self._structured_result, self)

    def invoke(self, prompt):
        self.text_invocations += 1
        self.last_text_prompt = prompt

        class FakeMessage:
            content = self._text_response

        return FakeMessage()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def fake_repo(tmp_path):
    """Cria um repositório fake com estrutura mínima para testar as tools."""
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()

    (tmp_path / "README.md").write_text("# Fake Repo\nProjeto de teste.\n")
    (tmp_path / ".env").write_text("APP_SECRET=super-secret\n")
    (tmp_path / "src" / "api.key").write_text("secret-key\n")
    (tmp_path / "src" / "main.py").write_text(
        "def circuit_breaker():\n    '''Implementa circuit breaker.'''\n    pass\n"
    )
    (tmp_path / "src" / "utils.py").write_text("def helper():\n    pass\n")
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "notes.md").write_text(
        "Notas do agente sobre circuit_breaker.\n"
    )

    # Arquivo grande para testar truncamento
    big_content = "\n".join(f"line {i}" for i in range(500))
    (tmp_path / "src" / "big_file.py").write_text(big_content)

    # Binário para testar exclusão de arquivos não-texto
    (tmp_path / "gateway").write_bytes(b"\x7fELF\x00circuit_breaker\x00" * 100)

    # Linha gigante para testar truncamento de matches do grep
    (tmp_path / "src" / "minified.json").write_text(
        '{"minified_payload": "' + "x" * 5000 + '"}\n'
    )

    return tmp_path
