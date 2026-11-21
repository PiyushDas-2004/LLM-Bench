# ADR 0002: Async runner with a fixed worker pool and per-model resilience controls

- Status: Accepted
- Date: 2026-11-21

## Context

An experiment is a grid of models x prompts x samples x trials; a modest config easily reaches
thousands of provider calls. Calls are I/O-bound, subject to provider rate limits, fail
transiently, cost money, and may be interrupted by the user. Results must be reproducible and an
interrupted run must be resumable without redoing finished work. The executor must be testable
without real time passing.

## Decision

The executor (`runner/executor.py`) is a single-process asyncio design:

- **Lazy, trial-major grid.** `PreparedExperiment.iter_items()` (`runner/prepare.py`) is a
  generator ordered trial -> sample -> model -> prompt, so a partial run covers every
  configuration evenly. `concurrency` workers (default 8) run in an `asyncio.TaskGroup` and pull
  from one shared iterator that skips keys already persisted. Memory stays flat for large grids.
- **Trial keys.** `model_id|prompt_id|sample_id|trial_index` is unique per run (enforced by
  `uq_trials_run_key`) and is the unit of checkpointing and resume.
- **Rate limiting** (`runner/rate_limit.py`). A `RateLimiter` combines a requests-per-minute
  and a tokens-per-minute `TokenBucket`; burst capacity is one second's worth (at least 1).
  The token estimate is prompt characters / 4 plus `max_tokens` (512 if unset), clamped to the
  bucket capacity. One limiter and one breaker are built per provider endpoint, keyed by
  `(provider, base_url, api_key_env)` in `_build_slots`, and shared by every model that uses
  that endpoint, because providers enforce limits per account and endpoint. Limits come from
  the first such model's `rate_limit` or the experiment-level one. (An earlier version built
  them per model spec; sharing exposed a float-precision livelock in `TokenBucket`, fixed with
  a tolerance and a 1 ms minimum wait.)
- **Retries** (`runner/retry.py`). Only `TIMEOUT`, `RATE_LIMITED`, `SERVER_ERROR` and `NETWORK`
  are retryable (`RETRYABLE_ERRORS` in `providers/base.py`); `AUTH`, `INVALID_REQUEST`,
  `CONTENT_FILTERED`, `REFUSAL` and `MALFORMED_OUTPUT` are not. The delay is full jitter,
  `U(0, min(max_delay_s, base_delay_s * 2**(attempt-1)))`, floored by a server `Retry-After`
  capped at 300 s. Each attempt is bounded by `asyncio.wait_for(timeout_s)`.
- **Circuit breaker** (`runner/circuit.py`). Opens after `failure_threshold` consecutive
  failures (default 8), waits `reset_timeout_s` (default 30), then admits a single half-open
  probe; success closes it, failure re-opens it. `INVALID_REQUEST` and `CONTENT_FILTERED` do not
  count. After `MAX_BREAKER_TRIPS = 3` openings the endpoint is abandoned for the run: every
  model on it stops retrying and its remaining items are skipped, leaving the run `partial`.
- **Budget** (`runner/budget.py`). Before a call the worker reserves an estimated cost
  (estimated input tokens plus the output ceiling, priced from `pricing.yaml`); reservations make
  `max_usd` safe under concurrency. After the call the reservation is settled with the actual
  cost (0 for failed calls). Exceeding `max_usd` or `max_requests` raises
  `BudgetExceededError`, which stops all workers gracefully with status `budget_exceeded`.
- **Checkpointing and resume.** Each finished trial (success or classified failure) is written
  immediately with `RunStore.add_trial`. `Runner.resume` rebuilds the config from the manifest,
  restricts samples to the manifest's sample ids, refuses if the evaluated dataset hash changed
  or a model is no longer runnable, and executes only missing keys.
- **SIGINT.** `cli._execute` installs `loop.add_signal_handler(SIGINT, runner.request_stop)`.
  Workers finish their in-flight trial, the run is marked `interrupted`, metrics are computed on
  what exists, and the CLI prints `llmbench resume <run_id>`.
- **Injectable clock** (`runner/clock.py`). Rate limiter, retry sleeps and breaker use a
  `Clock` protocol; tests pass `FakeClock`, whose `sleep` advances virtual time instantly.

## Consequences

Positive:

- Bounded memory and concurrency for any grid size; trial-major order makes partial results
  comparable across configurations.
- Interruption, budget stops and crashes lose at most the in-flight trials. The integration test
  `test_interrupt_and_resume_matches_uninterrupted_run` checks that a resumed run's outputs equal
  an uninterrupted run's on the mock provider.
- Every attempt is visible: `attempts`, `queue_wait_ms`, `error_class` are recorded per trial.
- Rate limiting, backoff and breaker logic are tested deterministically and fast.

Negative:

- Limits are per model spec, not per provider account: two models on the same key each get the
  full configured rate, so the experiment-level `rate_limit` should be set to the account limit
  divided by the number of models sharing it.
- Estimates are coarse (characters / 4). Unpriced models reserve 0, so `max_usd` does not protect
  them; `max_requests` counts trials, not retry attempts.
- A stop request is checked between trials only; a trial already in its retry loop continues
  until it succeeds or exhausts its attempts.
- `resume` verifies the dataset hash but re-reads prompt templates and the pricing file from
  disk without comparing them to the hashes stored in the manifest.
- Single process: throughput is bounded by one event loop. Horizontal scale needs several runs
  or the API behind a real queue (see ARCHITECTURE.md).

## Alternatives considered

- **Thread pool.** Simpler with blocking SDKs but heavier per task, harder to cancel, and the
  SDKs used here are async-native.
- **`asyncio.gather` over the materialised grid.** Trivial code, but memory grows with the grid
  and concurrency control moves into semaphores sprinkled across the call path.
- **SDK-level retries.** Hides attempts from the trial record and cannot share a breaker or a
  budget; see ADR 0008.
- **A third-party retry library (tenacity).** Capable, but Retry-After flooring, the breaker
  interplay and fake-clock testing were simpler to own in about 40 lines.
