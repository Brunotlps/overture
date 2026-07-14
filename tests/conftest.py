import pytest
from fastapi.testclient import TestClient

from app.main import app


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

    # Arquivo grande para testar truncamento
    big_content = "\n".join(f"line {i}" for i in range(500))
    (tmp_path / "src" / "big_file.py").write_text(big_content)

    return tmp_path
