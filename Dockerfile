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

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]