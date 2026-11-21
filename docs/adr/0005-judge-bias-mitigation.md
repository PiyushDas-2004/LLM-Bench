# ADR 0005: LLM-as-a-judge with structured output and explicit bias controls

- Status: Accepted
- Date: 2026-11-21

## Context

Lexical metrics miss correct paraphrases, so LLMBench offers an LLM judge
(`llm_judge_correctness`). LLM judges have documented failure modes: inconsistent scales, free
text that does not parse, preference for the first-presented answer (position bias), for longer
answers (length bias) and for their own outputs (self-preference), sampling noise, and unknown
agreement with humans. A judge score that silently carries these biases would undermine the
statistics built on it.

## Decision

Implemented in `src/llmbench/judges/`:

- **Rubrics with anchors** (`rubric.py`, e.g. `rubrics/correctness.yaml`): named, weighted
  criteria, an integer scale (default 1-5), a `pass_threshold`, and per-score anchor
  descriptions with optional examples. The rubric is validated (threshold and anchors inside the
  scale). The system prompt tells the judge to ignore length and style unless the rubric says
  otherwise and to treat candidate text as data, not instructions.
- **Structured output with validation and repair** (`judge.py`): every call requests JSON
  (`json_mode`), the reply is parsed with `extract_json` and validated against a JSON Schema
  (`Draft202012Validator`; integer `score` within the scale, or `winner` in `A`/`B`/`tie`). On
  failure the judge is re-prompted with the validation error, up to `max_repairs` times
  (default 2); a still-invalid reply yields no score and an error, never a guessed score.
- **Temperature 0 by default** (`JudgeSpec.params` and `LLMJudge` default to
  `{"temperature": 0.0}`).
- **Position swap** for pairwise judging (`LLMJudge.pairwise`, `swap_order=True` by default):
  both orders are judged and mapped back to the inputs; if they disagree the result is a tie
  with `position_consistent=False`.
- **Length bias reporting**: `length_bias` computes the Spearman correlation between answer
  length and normalised score; the run summary recomputes it from the persisted per-trial
  `details.length` and shows it in the report.
- **Self-preference warning**: `build_judge` (`factory.py`) logs `judge_is_candidate` when the
  judge's provider and model equal a candidate's; the summary also flags `candidate_overlap`
  when the judge model name appears among the run's models.
- **Self-consistency**: `samples > 1` collects independent judgments; the score is their mean
  and `self_consistency` is the modal share. Repeat calls use distinct `trial_index` values.
- **Calibration against humans** (`calibration.py`, `llmbench judge calibrate`): runs the judge
  on a labelled JSONL set and reports accuracy, Cohen's kappa (pass/fail), quadratic-weighted
  kappa and Spearman on scores when human scores exist, the confusion matrix and mean
  self-consistency. `--attach <run>` stores the report as a run artifact shown in the report.
- **Audit trail**: every judge call, including repairs and provider errors, is written to the
  `judge_calls` table with prompt, raw output, parsed result, tokens, cost, latency and error
  (`run_id` is null for calibration calls).
- Failed candidate trials score 0 without calling the judge; a judge failure leaves the value
  null, so it is excluded rather than counted as wrong.

## Consequences

Positive:

- Judge scores are reproducible to the extent the provider honours temperature 0, parse
  failures are explicit, and every number can be traced to the exact prompt and reply.
- Biases are measured and surfaced rather than assumed away; calibration gives a number
  (kappa) to decide whether a judge is fit for gating.
- The mock provider simulates judging (token-overlap scores, malformed JSON, `position_bias`),
  so all of this is tested offline.

Negative:

- Judge calls bypass the runner: no rate limiter, retry policy, circuit breaker, cache or
  budget applies, and judge cost is not counted against `max_usd`. A provider error fails that
  judgment immediately.
- Position swap applies only to pairwise judging, which is available through the Python API;
  no built-in metric or CLI command currently uses it. Pointwise scoring has no order to swap.
- Length bias is reported, not corrected. Per-item self-consistency is returned by the judge but
  not persisted in metric details, so for normal runs it is only visible in calibration reports.
- The self-preference check compares names only; a fine-tune or alias of the same base model
  is not detected.
- With `samples > 1` and repairs, one judged trial can cost several calls.

## Alternatives considered

- **Free-text judge output parsed with regex.** Fragile; silently misreads scores.
- **Logprob-weighted scores.** Not uniformly available across the supported providers.
- **Pairwise-only judging.** Strong for A/B but needs a partner answer for every item and does
  not give an absolute score for gating.
- **Discard length-biased judges automatically.** No principled threshold; reporting the
  correlation leaves the decision to the user with calibration data at hand.
