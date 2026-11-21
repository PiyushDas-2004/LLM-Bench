# CLAUDE.md — working notes for LLMBench

LLMBench is an evaluation, benchmarking and regression-testing framework for
non-deterministic LLM and RAG systems. Python 3.11+, `src/` layout, package `llmbench`.

## Dev commands

```bash
uv sync                      # create .venv with all extras + dev tools (pip: pip install -e ".[all]")
make lint                    # ruff check + ruff format --check
make type                    # mypy --strict on src/
make test                    # pytest (offline, mock provider) with the 85% coverage gate
make security                # bandit + pip-audit
make bench                   # offline mock benchmark: llmbench run examples/quickstart.yaml --yes
pytest -m live -o addopts="" # provider contract tests; need OPENAI_API_KEY / GEMINI_API_KEY
```

## Architecture (dependency direction: top depends on bottom)

```
cli.py / api/            entry points (typer CLI, FastAPI)
reporting/               summaries, rich tables, HTML/Markdown/JSON/CSV/Parquet
tracking/                SQLAlchemy models, alembic migrations, repository, compare, gates, drift
runner/                  async executor, rate limiter, retry, circuit breaker, budget, cache, plan
metrics/ judges/ rag/    metric registry + built-ins, LLM-as-judge, RAG pipeline + metrics
stats/                   bootstrap (percentile/BCa/cluster), paired tests, effect sizes, power
tasks/                   task types (qa, extraction, classification, safety, rag, generic)
providers/               Provider protocol, error taxonomy, mock + openai/gemini/openai_compat
prompts/ datasets/       sandboxed Jinja2 templates, dataset schema/hashing/sampling
config/ plugins/         pydantic config + settings + pricing, entry-point registries
```

Rules:
- Core code never imports provider SDKs; only `providers/openai_provider.py` and
  `providers/gemini_provider.py` may (enforced by ruff TID251).
- Every test runs offline against `MockProvider`; real-API tests carry `@pytest.mark.live`.
- Never hardcode model names or prices; models live in experiment YAML, prices in `pricing.yaml`.
- Never commit secrets, `.env`, or `*.db` files. Keys come only from the environment.
- Numbers in docs must come from real runs; label mock-provider numbers as illustrative.

## Useful facts
- Run store defaults to `sqlite:///.llmbench/llmbench.db` (override with `--db` / `LLMBENCH_DB_URL`).
- `LLMBENCH_PLUGINS=mod1,mod2` imports custom metric/provider/task modules at CLI startup.
- The mock provider's knobs live in `providers/mock.py`; `difficulty_seed` fixes per-sample
  difficulty, `mock_seed` changes only sampling noise (use it to simulate a re-run).
- Golden Markdown reports live in `tests/golden/`; regenerate with `UPDATE_GOLDEN=1 pytest tests/integration/test_reporting.py`.
- The CI eval-gate baseline is `baselines/gate-demo.json` (`llmbench export <run> --output ...`).
- Rate limiters and circuit breakers are shared per provider endpoint (provider, base_url, api_key_env).

## Conventions
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`), one module per commit.
- Type hints everywhere, `mypy --strict` clean, ruff clean, docstrings on public APIs.
- No TODOs, no commented-out code. Record significant decisions as ADRs in `docs/adr/`.
