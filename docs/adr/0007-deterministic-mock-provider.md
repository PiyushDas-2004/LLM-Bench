# ADR 0007: A deterministic, oracle-driven mock provider

- Status: Accepted
- Date: 2026-11-21

## Context

LLMBench must be testable and demonstrable without API keys or network access: the unit and
integration tests, the quickstart, the `llmbench init` scaffold and the repository's own CI
gate all need a provider. A trivial echo stub would not exercise what LLMBench is for:
variability across trials, a controllable quality level, failures that trigger retries and the
circuit breaker, judge parsing, RAG citations and safety probes. Resume must also be checkable,
which requires the same trial to produce the same output on re-execution.

## Decision

`providers/mock.py` implements `MockProvider`, registered as the `mock` provider:

- **Oracle metadata.** The runner attaches `metadata["oracle"]` (from `Task.oracle`: sample id,
  reference, difficulty, task name, sample metadata) to each request. Real adapters never send
  `metadata`; the mock uses it to decide whether to answer correctly and to build plausible
  wrong answers (another label, a perturbed number, a broken JSON field, a filler word).
- **Decisions are a pure function of inputs.** Every random draw comes from
  `random.Random` seeded with SHA-256 of a tuple built from `mock_seed`, the model name, the
  rendered prompt text, the trial index and, for failures and latency, a per-key attempt count.
- **Temperature 0 means identical trials.** When the effective temperature is 0 the trial index
  is dropped from the key, mimicking greedy decoding.
- **Quality model.** P(correct) is a logistic function of `accuracy`, the sample's
  `difficulty` (easy +1, hard -1 on the logit) and a per-sample offset drawn with
  `difficulty_seed` and scaled by `sample_spread`.
- **Two seeds.** `difficulty_seed` fixes which samples are hard for a model;
  `mock_seed` changes only the sampling noise. Changing `mock_seed` simulates a re-run of the
  same system; changing `accuracy` simulates a real quality change. The gate examples
  (`examples/gate/noise.yaml` vs `regression.yaml`) rely on exactly this split.
- **Failure injection.** `failure_rate` raises `ProviderError` with a class drawn from
  `error_mix` (TIMEOUT / RATE_LIMITED / SERVER_ERROR); simulated 429s carry `retry_after_s`.
  Also `refusal_rate`, `malformed_rate`, log-normal `latency_ms` / `latency_sigma` (slept only
  as `time_scale` x latency, default 0), and TTFT when `stream` is set.
- **Behaviours.** Judge mode (`metadata["mock_judge"]`): pointwise scores from token-F1 against
  the reference plus noise, pairwise winners with optional `position_bias`, malformed JSON for
  repair tests. RAG: cites the numbered context blocks containing gold evidence, abstains when
  evidence is absent or the question is unanswerable. Safety: obeys an injected instruction
  (emits the canary) with probability `injection_susceptibility`.
- Results carry `provider_metadata={"simulated": True}`; embeddings come from the deterministic
  `HashEmbedder`.

## Consequences

Positive:

- CI and every test run keyless and offline (pytest deselects `live` tests by default).
- Outputs are reproducible across machines and processes; an interrupted and resumed run
  produces the same trial outputs as an uninterrupted one, which the integration tests assert.
- Statistical machinery is validated against known ground truth: a seeded regression must be
  detected, a seed-only change must not.

Negative and risks:

- Mock numbers can be mistaken for measurements of real models. Mitigations: the HTML report
  shows a "Mock provider" banner and the Markdown run summary a matching note whenever any
  configuration uses `mock`; example configs and `CLAUDE.md` require mock numbers in docs to be
  labelled illustrative. The gate Markdown (`gate.md`) does not carry the banner, so CI
  summaries from mock runs should say so in the workflow step name.
- The attempt counter is per provider instance and per key. At temperature 0 all trials of a
  prompt share one key, so with concurrency > 1 the latency and failure draws a trial receives
  depend on execution order; the answer text does not.
- Difficulty is seeded per (sample, model name), so two differently named mock models do not
  share a difficulty profile.
- The oracle is excluded from the cache key (ADR 0003); editing a reference without changing
  the input can replay a stale mock answer.
- Realism is limited: the mock encodes a simple quality model and cannot stand in for contract
  tests, which exist separately as `@pytest.mark.live` tests against real APIs.

## Alternatives considered

- **Recorded fixtures (VCR-style cassettes).** Realistic text, but fixed to recorded prompts,
  break whenever a prompt changes, cannot produce new trials, and need keys to re-record.
- **A small local model.** Real generation, but slow, heavy in CI, and not deterministic across
  hardware without care.
- **Seeding Python's global RNG.** Order-dependent under concurrency; hashing the inputs makes
  each decision independent of scheduling.
- **No oracle (random answers).** Every metric would sit near chance and regression detection
  could not be tested against a known effect.
