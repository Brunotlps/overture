# Overture Documentation

This directory documents the current implementation of Overture: a FastAPI service
that answers questions about Git repositories through a LangGraph ReAct agent.

The README at the repository root is the entry point. These documents hold the
details that are useful when changing, operating, or evaluating the system.

## Documents

- [Getting started](getting-started.md) - local setup, environment variables, Docker, and common commands.
- [API](api.md) - public HTTP endpoints and request/response contracts.
- [Architecture](architecture.md) - runtime components, graph flow, repository tools, and known boundaries.
- [Operations](operations.md) - startup repository provisioning, deploy, CI/CD, logs, and troubleshooting.
- [Testing and eval](testing-and-eval.md) - pytest coverage, linting, and the LLM-backed eval harness.
- [Security](security.md) - API key authentication, tool guardrails, logging controls, and remaining risks.

## Evidence Policy

The documentation describes behavior that is present in the codebase today. When
something is planned or intentionally limited, it is labeled that way and linked to
the relevant source where possible.

Primary evidence used:

- Application code under `app/`.
- Tests under `tests/`.
- Local eval harness under `eval/`.
- Runtime and deployment files: `Dockerfile`, `fly.toml`, `.github/workflows/ci.yml`.
- Git history and GitHub issues/PRs available during the documentation review.

## Current Scope

Overture supports:

- a default target repository configured by `APP_REPO_PATH`;
- optional startup clone from `APP_REPO_GIT_URL`;
- an optional curated portfolio repo list loaded from `portfolio_repos.yaml`;
- authenticated `/ask` and `/repos`;
- public `/health`;
- in-memory conversation threads via LangGraph `MemorySaver`;
- repository tools exposed to the LLM: `list_files`, `read_file`, and `grep_repo`.

It does not currently support arbitrary request-time repository registration,
persistent conversation memory, metrics/tracing, per-client API keys, rate limiting,
or semantic search.
