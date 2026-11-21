# ADR 0003: Response cache keyed per trial, including model params and base URL

- Status: Accepted
- Date: 2026-11-21

## Context

Re-running an unchanged experiment (to regenerate a report, tweak a metric, or recover from a
crash in evaluation) should not pay for the same provider calls twice. A naive prompt-level cache
breaks the statistics, though: LLMBench runs each prompt several times (`trials`) precisely to
measure variability, and if trial 2 returned trial 1's cached text every consistency metric,
pass@k and cluster-bootstrap interval would be computed over copies of one sample.

## Decision

`runner/cache.py` implements a read-through cache stored in the run database
(`cache_entries` table, ADR 0001). The key is the SHA-256 of a canonical JSON payload
(`canonical_json`: sorted keys, compact separators) containing:

- `provider`: the provider name (`mock`, `openai`, `openai_compat`, `gemini`, or a plugin);
- `model_config`: the model spec's full `params` mapping and its `base_url`;
- `model`: the provider-side model name;
- `params`: the normalised `GenerationParams` (temperature, top_p, max_tokens, seed, stop,
  json_mode, stream);
- `messages`: the rendered system and user messages;
- `trial_index`.

Rules:

- **Trial index in the key.** Repeated trials stay independent samples. Re-running the same
  config replays the same trial 0..n-1 outputs, which keeps re-runs identical while still
  having n distinct draws per sample.
- **Model params and base URL in the key.** Added by commit
  "fix(runner): include model params and base_url in the response-cache key". Before it, only the
  normalised `GenerationParams` were hashed, so provider-specific knobs that live in `params`
  but not in `GenerationParams` were invisible to the key. A test changing the mock provider's
  `accuracy` hit stale entries and reported the old behaviour. The same flaw would have let two
  OpenAI-compatible servers serving the same model name under different `base_url`s share
  entries. `test_cache_key_includes_trial_index_and_params` now covers this.
- **Only successful results are cached** (`ResponseCache.put` returns early unless
  `result.ok`). Transient failures are retried on the next run instead of being replayed.
- **Cached trials** are recorded with `cached=True`, one attempt and zero queue wait. Their
  `cost_usd` is computed from the original call's token counts with the run's pricing table, so
  per-configuration cost reports stay comparable. They skip budget reservation and are not added
  to `BudgetGuard.spent_usd` (no money is spent), so a fully cached re-run reports
  `spent $0.0000` while `runs.total_cost_usd` still sums the nominal trial costs.
- Caching is on by default; it can be disabled per experiment (`cache: false`), per invocation
  (`--no-cache` on `run`, `resume`, `rerun`), and cleared with `llmbench cache clear`. A run's
  manifest records `cache_enabled`, which `resume` honours.

## Consequences

Positive:

- Re-running an identical config is free and instant, and produces identical trial outputs
  (`test_cache_makes_rerun_free`).
- Trials remain statistically independent draws; cached and uncached runs have the same shape.
- Any parameter a user can set on a model now invalidates its entries.

Negative:

- The key uses the requested model name, not the version the provider actually served. If a
  provider re-points an alias, cached entries keep returning the old model's outputs. Drift
  monitoring and baseline refreshes should use `--no-cache`.
- `GenerationRequest.metadata` is not part of the key. Real providers never see it, but the
  mock reads its `oracle`: editing a sample's reference without changing its input can replay an
  answer computed against the old reference.
- Hashing the whole `params` mapping also includes knobs that cannot change the output (for
  example `timeout_s`), so changing them causes cache misses.
- The cache grows without bound until cleared; `cache_clear(older_than=...)` exists in the store
  but the CLI exposes only a full clear.

## Alternatives considered

- **Prompt-level key (no trial index).** Maximises hits but collapses repeated trials into one
  sample, invalidating consistency metrics and confidence intervals.
- **Hash only `GenerationParams`.** The original implementation; it missed provider-specific
  parameters and `base_url`, as described above.
- **Cache failures too.** Would make reruns faster on flaky providers but would freeze transient
  errors into results.
- **External cache (Redis, on-disk key-value store).** Another moving part; the run database
  already provides durable storage and is shared by CLI and API.
