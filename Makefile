.DEFAULT_GOAL := help
PY ?= python
SRC := src tests

.PHONY: help install lint format type test test-live cov security bench report docs docker docker-run clean check-all

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install the package with dev tooling (uv if available, else pip)
	@if command -v uv >/dev/null 2>&1; then uv sync; else $(PY) -m pip install -e ".[all]" pytest pytest-asyncio pytest-cov hypothesis httpx2 ruff mypy bandit pip-audit types-PyYAML types-jsonschema pre-commit; fi

lint: ## Ruff lint and format check
	ruff check $(SRC)
	ruff format --check $(SRC)

format: ## Auto-format and fix lint
	ruff format $(SRC)
	ruff check --fix $(SRC)

type: ## mypy --strict on src/
	mypy src

test: ## Run the offline test suite with coverage gate
	pytest --cov --cov-report=term-missing

test-live: ## Run provider contract tests against real APIs (needs keys)
	pytest -m live -o addopts=""

security: ## bandit + pip-audit
	bandit -q -r src -c pyproject.toml
	pip-audit --skip-editable

bench: ## Run the offline mock benchmark
	llmbench run examples/quickstart.yaml --yes

report: ## Render HTML report for the most recent run
	llmbench report latest --format html --output reports/latest.html

docs: ## Build the mkdocs site (optional dependency)
	mkdocs build --strict

docker: ## Build the Docker image
	docker build -t llmbench:local .

docker-run: ## Run the mock benchmark inside the container
	docker run --rm llmbench:local run examples/quickstart.yaml --yes

check-all: lint type test security ## Everything CI runs

clean: ## Remove caches and build outputs
	rm -rf build dist .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov
