# Architecture Decision Records

Each record captures one significant design decision in LLMBench: the context that forced
it, what was decided, its positive and negative consequences, and the alternatives that were
rejected. Every statement in a record refers to code that can be checked; module paths are
relative to `src/llmbench/` unless stated otherwise.

| ADR | Title | Status | Main code |
|---|---|---|---|
| [0001](0001-sqlite-run-store.md) | SQLite run store via SQLAlchemy 2.0 and Alembic | Accepted | `tracking/db.py`, `tracking/store.py`, `tracking/migrate.py`, `tracking/bundle.py` |
| [0002](0002-async-runner.md) | Async runner with a fixed worker pool and per-model resilience controls | Accepted | `runner/executor.py`, `runner/rate_limit.py`, `runner/retry.py`, `runner/circuit.py`, `runner/budget.py`, `runner/clock.py` |
| [0003](0003-trial-level-cache-key.md) | Response cache keyed per trial, including model params and base URL | Accepted | `runner/cache.py` |
| [0004](0004-bootstrap-design.md) | Cluster bootstrap for intervals, paired per-sample tests for comparisons | Accepted | `stats/bootstrap.py`, `stats/compare.py`, `stats/paired.py` |
| [0005](0005-judge-bias-mitigation.md) | LLM-as-a-judge with structured output and explicit bias controls | Accepted | `judges/judge.py`, `judges/metrics.py`, `judges/factory.py`, `judges/calibration.py` |
| [0006](0006-plugin-system.md) | Typed registries with lazy entry-point discovery for plugins | Accepted | `plugins/registry.py`, `metrics/base.py`, `providers/registry.py`, `tasks/base.py` |
| [0007](0007-deterministic-mock-provider.md) | A deterministic, oracle-driven mock provider | Accepted | `providers/mock.py` |
| [0008](0008-chat-completions-and-sdk-retries.md) | Chat Completions for OpenAI-style servers, google-genai for Gemini, SDK retries off | Accepted | `providers/openai_provider.py`, `providers/gemini_provider.py`, `providers/_http.py` |
| [0009](0009-web-dashboard.md) | Static web dashboard with demo (CI JSON) and live (REST API + CORS) modes | Accepted | `frontend/`, `scripts/build_site.sh`, `api/app.py` |

## How the records relate

- 0001 (storage) underpins 0002 (per-trial checkpointing and resume) and 0003 (the cache
  lives in the run database).
- 0002 owns retries, which is why 0008 disables SDK-level retries.
- 0004 defines the statistics that the CI gate in [../CI_EVAL_GATE.md](../CI_EVAL_GATE.md)
  builds on; 0007 makes that gate testable offline with a known seeded regression.
- 0005 and 0006 describe extension points (judge rubrics, plugins) listed in
  [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Format

Every record uses the same sections:

1. Title (`# ADR NNNN: ...`)
2. Status (`Proposed`, `Accepted`, `Superseded by NNNN` or `Deprecated`)
3. Date (ISO 8601)
4. Context
5. Decision
6. Consequences, split into positive and negative
7. Alternatives considered

## Adding a record

1. Copy the most recent record to `NNNN-short-slug.md` with the next free number.
2. Write it in the past tense for facts and the present tense for the decision; name the
   modules involved so a reader can verify each claim.
3. Do not quote benchmark numbers unless they come from a recorded run; label numbers from
   the mock provider as illustrative.
4. Add a row to the table above.
5. When a decision is replaced, keep the old record, set its status to
   `Superseded by NNNN`, and link both ways.
