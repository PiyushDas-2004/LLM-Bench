# ADR 0001: SQLite run store via SQLAlchemy 2.0 and Alembic

- Status: Accepted
- Date: 2026-11-21

## Context

Every trial LLMBench executes costs time and, against real providers, money. Results must
survive crashes and Ctrl-C, be queryable later for comparison, drift detection and gating, and
carry enough provenance (config, manifest, dataset hash) to explain where a number came from.
The tool is mostly run on a laptop or in a short-lived CI job, where running a database server
is friction. The API service (`llmbench serve`) and the CLI should share one storage layer.

## Decision

- Persist everything in a relational store accessed through SQLAlchemy 2.0 ORM models
  (`tracking/db.py`): `runs`, `trials`, `metric_values`, `judge_calls`, `baselines`,
  `cache_entries` and `artifacts`. Flexible payloads (config, manifest, tags, parsed output,
  metric details, cached responses) use the generic `JSON` column type.
- Default URL is `sqlite:///.llmbench/llmbench.db` (`config/settings.py`); `make_engine`
  creates the parent directory. Any SQLAlchemy URL can be supplied with `LLMBENCH_DB_URL` or
  the global `--db` option, so Postgres is possible without code changes.
- On SQLite, a connect listener sets `PRAGMA foreign_keys=ON`, `journal_mode=WAL` and
  `synchronous=NORMAL`.
- The schema is owned by Alembic migrations packaged inside the wheel
  (`tracking/migrations/`, currently `0001_initial_schema`). `tracking/migrate.py` builds the
  Alembic config programmatically, so no `alembic.ini` is needed, and `RunStore.__init__`
  upgrades to `head` on every open. `render_as_batch=True` keeps SQLite `ALTER` migrations
  workable. `test_migrations_match_models` asserts the ORM models and migrations never drift.
- `RunStore` (`tracking/store.py`) is the only repository. It uses short, synchronous
  sessions (`session.begin()` per operation, `expire_on_commit=False`).
- `add_trial` writes and commits one trial at a time; the trial is the checkpoint unit that
  `resume` relies on (`completed_trial_keys`).
- Writes are issued synchronously from the asyncio event loop (runner workers, API job queue).
  This is accepted because a run spends almost all of its wall time awaiting provider calls,
  and a local SQLite commit in WAL mode with `synchronous=NORMAL` does not fsync the database
  on every commit. Moving writes to a thread would add a second concurrency domain for little
  gain at the concurrency levels the runner targets.
- Portable run bundles (`tracking/bundle.py`, format `llmbench-run/1`) export a run's
  metadata, manifest, trials and metric values to JSON and import them into another store.
  CI commits a bundle as the stored baseline instead of sharing a database file
  (see `docs/CI_EVAL_GATE.md`).

## Consequences

Positive:

- Zero-setup persistence: the first `llmbench run` creates a working database.
- Crash safety at trial granularity; an interrupted run resumes exactly where it stopped.
- One schema serves CLI, API, reports, comparisons and gates; migrations are versioned and
  tested, and upgrades happen automatically on open.
- Bundles make baselines reviewable in a pull request and independent of any database.

Negative:

- Synchronous commits briefly block the event loop. With a remote database (Postgres over a
  network) each commit adds a round trip on the loop; with the mock provider and
  `time_scale: 0` the database is the bottleneck. Neither matters for correctness.
- SQLite allows a single writer. Several processes writing to one file (for example two CI
  jobs on a shared volume) will contend; use separate files or Postgres.
- Postgres support is by construction (generic types, no SQLite-only SQL in the store), but
  the offline test suite only exercises SQLite, and no Postgres driver is a core dependency.
- Auto-upgrade on open means an older LLMBench cannot be trusted to open a database that a
  newer version migrated.
- Bundle export stores the dataset path and `base_dir` relative to the project root (the
  working directory at export time) and `llmbench import --base-dir` (default `.`) resolves them
  again, so export and import must both run from the project root. A path outside that root
  keeps the exporting machine's absolute path. Gating does not need the path; re-evaluating an
  imported run does.

## Alternatives considered

- **JSONL files per run.** Simple and diff-friendly, but every comparison would re-implement
  indexing and joins, and concurrent appends need their own locking. Bundles keep the
  diff-friendly property where it matters (committed baselines).
- **DuckDB or Parquet as the primary store.** Strong for analytics, weaker for many small
  transactional writes from concurrent workers. Parquet remains an export format.
- **An async driver (`aiosqlite`, SQLAlchemy asyncio).** Removes loop blocking but adds a
  dependency and async session management across every caller (CLI, reports, stats) for a
  cost that is small relative to provider latency.
- **Require Postgres.** Better multi-writer behaviour, but defeats the offline, keyless,
  single-command workflow that tests, the quickstart and CI depend on.
