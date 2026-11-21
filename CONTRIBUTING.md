# Contributing to LLMBench

Thanks for helping. This guide covers the local workflow and the bar a change must meet.

## Setup

```bash
git clone https://github.com/PiyushDas-2004/LLMBench.git && cd LLMBench
uv sync                 # or: python -m venv .venv && pip install -e ".[all]" + dev tools (see Makefile)
uv run pre-commit install
```

## Before opening a pull request

```bash
make lint type test security
```

- `ruff check` and `ruff format --check` are clean.
- `mypy --strict src` is clean.
- `pytest` passes offline with coverage >= 85% (CI enforces this).
- New behaviour has tests that run against the mock provider. Tests that need real APIs
  are marked `@pytest.mark.live` and skip cleanly without keys.
- Public functions and classes have docstrings and type hints.
- No TODO placeholders or commented-out code.

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `test:`, `docs:`, `ci:`, `chore:`, `refactor:`. One logical change per commit.
- Core modules never import provider SDKs (ruff rule `TID251` enforces this); SDK code
  lives in `src/llmbench/providers/*_provider.py`.
- Never hardcode model names or prices. Models belong in experiment YAML; prices in a
  pricing YAML with an `as_of` date.
- Numbers in docs must come from runs you actually executed, labelled with the provider
  (mock numbers are always marked illustrative).
- Significant design decisions get an ADR in `docs/adr/`.

## Adding a metric, provider or task

See "Extending LLMBench" in the README. Built-ins register in their module; external
packages use the `llmbench.metrics`, `llmbench.providers` or `llmbench.tasks` entry-point
groups.

## Reporting bugs and security issues

Open an issue with the command, config (secrets removed) and full output. Report
security vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
