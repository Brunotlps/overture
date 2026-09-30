# Testing and Eval

Overture has two quality loops:

- deterministic local tests with pytest and fakes;
- a manual LLM-backed eval harness under `eval/`.

## Pytest

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

The test suite uses fake LLM objects and temporary repositories, so normal pytest
runs do not need LLM credentials.

Observed coverage by area:

| Area | Evidence |
| --- | --- |
| Endpoint contract and `/ask` ReAct loop | `tests/test_health.py` |
| API key fail-closed behavior | `tests/test_security.py` |
| Tool path traversal, ignored dirs, sensitive paths, binary handling, line ranges, listing pages, output caps, bounded memory | `tests/test_tools.py` |
| ReAct graph nodes, routing, guardrails, unknown tools | `tests/test_graph.py` |
| Per-request answer language prompt/fallbacks and API validation | `tests/test_language.py` |
| Conversation memory, per-turn budget reset, and summarization flow | `tests/test_memory.py`, `tests/test_summarization.py` |
| Thread retention: latest checkpoint only, idle TTL, LRU thread cap, in-flight protection, settings validation | `tests/test_retention.py` |
| Tool-call pairing across compaction, budget guardrail, tool errors, and resumed threads | `tests/test_tool_protocol.py` |
| Prompt-injection trust boundary, malicious fixtures, summary carryover, and adversarial rubric | `tests/test_prompt_injection.py`, `tests/test_adversarial_eval.py` |
| Optional semantic search indexing limits, partial coverage, batching, ranking, graceful failure, and tool registration | `tests/test_semantic_search.py`, `tests/test_agent_tools.py` |
| Structured logs, private mode, error boundaries, and diagnostic limits | `tests/test_observability.py`, `tests/test_privacy.py` |
| Startup repo clone behavior | `tests/test_repo.py` |
| Curated repo YAML and registry | `tests/test_portfolio.py`, `tests/test_repo_registry.py`, `tests/test_repos_endpoint.py`, `tests/test_ask_repo_id.py` |

## Lint

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run ruff check
```

CI runs the same lint command.

## CI

`.github/workflows/ci.yml` has two jobs:

- `test`: installs with `uv sync --locked`, then runs pytest and ruff.
- `deploy`: runs `flyctl deploy --remote-only` on pushes to `main` after `test` succeeds.

The workflow uses pinned GitHub Action revisions.

Recent full-suite result reported in PR #33: `118 passed` and `ruff check` clean.

## Eval Harness

Run:

```bash
uv run python -m eval.run
```

The eval harness:

- builds a fresh ReAct graph per case and repeat, retaining a checkpoint only
  between turns of the same case;
- points it at `eval/fixture_repo`;
- checks conclusion, expected tools, retrieved files, cited files, and explicit
  fact patterns independently;
- stores the answer, model, prompt hash, semantic-search setting, and a hash of
  the fixture files in an optional JSON report.

Cases in `eval/cases.py` cover behavior, references, a conceptual question with
no literal match, missing information, and a two-turn continuation. The fact
patterns are transparent smoke checks: a passing pattern is not proof that the
whole answer is correct. `answered` counts conclusions, not factual accuracy.
Review a sample of answers and citations when interpreting a run.

For a descriptive lexical/semantic comparison, run both configurations with
the same fixture and save their reports:

```bash
uv run python -m eval.run --repeats 3 --output /tmp/eval-lexical.json
APP_SEMANTIC_SEARCH_ENABLED=true uv run python -m eval.run \
  --repeats 3 --output /tmp/eval-semantic.json \
  --compare /tmp/eval-lexical.json
```

The comparison reports observed counts and configurations; repeated calls do
not establish statistical improvement. Retrieval is guided by agent tools,
not a mandatory RAG pipeline. This command is opt-in and calls a real LLM;
CI runs only the offline scoring tests.

## Prompt-injection evaluation

The offline tests verify that repository text stays in tool messages, and that
conversation summaries are passed as labeled historical context outside the
system message. They also exercise malicious instructions in a synthetic
`README.md` and source file, plus the real compaction path with a fake model.

To observe a real model against those fixtures, run:

```bash
uv run python -m eval.run --adversarial --output /tmp/eval-adversarial.json
```

This opt-in run calls the configured LLM. Its cases cover instructions in a
README, a tool result, and a follow-up after summarization. The JSON report
separately records task deviation, attempted access to a blocked `.env` path,
and contamination of the later answer. It also checks that the expected source
was read. Inspect the recorded answers and tool calls: regex checks can miss
paraphrases or flag harmless discussion of attack text. Passing checks do not
establish immunity to prompt injection.

## Known Gaps

- Pytest verifies graph mechanics with fakes, not real model answer quality.
- Prompt-injection fixtures and rubric cover selected attacks, not all possible
  model behavior or exfiltration routes.
- The eval fixture is small; PR #27 reports a positive semantic-search signal on the
  "money" conceptual case, but not strong statistical evidence.
- No coverage report is generated by default.
