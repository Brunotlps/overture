# ---- Stage 1: builder ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Cache de dependências: copia só o que define dependências primeiro
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Agora copia o código e instala o próprio projeto
COPY app/ ./app/
RUN uv sync --locked --no-dev

# ---- Stage 2: runtime ----
FROM python:3.12-slim-bookworm

# git é usado no startup para clonar o repositório alvo (APP_REPO_GIT_URL)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY portfolio_repos.yaml ./

ENV PATH="/app/.venv/bin:$PATH"

# Pré-clona os repos do portfólio na imagem: ensure_repo() trata diretório
# não-vazio como pronto, então o startup não repete os clones e não estoura
# o timeout do proxy da Fly no cold start (scale-to-zero).
# Nota: o conteúdo dos repos congela no build; um novo deploy atualiza tudo.
RUN python - <<'PYEOF'
import subprocess
from pathlib import Path

import yaml

repos = yaml.safe_load(Path("portfolio_repos.yaml").read_text())["repos"]
for repo in repos:
    dest = Path("/data/repos") / repo["repo_id"]
    subprocess.run(
        ["git", "clone", "--depth", "1", repo["git_url"], str(dest)],
        check=True,
    )
PYEOF

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]