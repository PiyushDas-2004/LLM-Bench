# ADR 0004: Cluster bootstrap for intervals, paired per-sample tests for comparisons

- Status: Accepted
- Date: 2026-11-21

## Context

LLM outputs vary between trials of the same input, and inputs vary in difficulty. Benchmarks
are usually small (tens to hundreds of samples) and metrics are often binary. Treating each
trial as an independent observation understates uncertainty, because trials of one sample are
correlated. Comparisons between two configurations are evaluated on the same samples, so the
pairing should be exploited. A CI gate built on these numbers must not fail on noise, and when
many metrics are compared at once the family-wise error must be controlled.

## Decision

Statistics live in `src/llmbench/stats/` and are implemented directly on numpy and scipy:

- **Per-configuration intervals** use a two-stage cluster bootstrap
  (`cluster_bootstrap_ci` in `bootstrap.py`): stage 1 resamples samples with replacement, stage 2
  resamples trials within each chosen sample. The estimate is the mean of per-sample means, so
  every sample weighs equally regardless of how many trials succeeded. This is used by run
  summaries (`reporting/summary.py`, 1000 resamples) and by comparisons (2000 resamples).
- **Paired comparisons** (`compare_metric` in `compare.py`) align the two runs on common sample
  ids and reduce each side to per-sample means. The default test (`choose_test`) is:
  - McNemar (`mcnemar_test`) when the metric is binary and both sides have exactly one trial per
    sample (exact binomial below 25 discordant pairs, continuity-corrected chi-square above);
  - otherwise a sign-flip permutation test on the per-sample differences
    (`paired_permutation_test`): exact enumeration up to 16 non-zero differences, else 10,000
    Monte Carlo flips with the +1 correction so p is never 0.
  Wilcoxon and a paired bootstrap test are available by name.
- **Delta interval**: a paired bootstrap over per-sample differences (`paired_bootstrap`).
- **Effect sizes**: Cliff's delta for binary metrics, Cohen's d_z otherwise (`effect.py`); the
  minimum detectable effect (`power.py`, normal approximation, power 0.8) is reported, with a
  warning below 30 paired samples.
- **Multiple comparisons**: p-values within one configuration pair are adjusted across metrics
  with Holm by default (`correction.py`); Benjamini-Hochberg and `none` are selectable.
- **Verdict** (`_verdict`): `improved` or `regressed` only when the adjusted p < alpha AND the
  delta CI excludes 0; otherwise `no significant change`; fewer than 2 aligned samples gives
  `insufficient data`. The gate (`tracking/gates.py`) adds a practical-size threshold on top.
- **BCa** (`bca_ci`, Efron 1987, jackknife acceleration, percentile fallback when degenerate)
  and a percentile bootstrap are part of the public `llmbench.stats` API. The run pipeline does
  not call BCa; the RAG sweep uses the percentile interval over per-question values.
- All resampling uses `numpy.random.default_rng` with fixed seeds, so a comparison of the same
  two stored runs is reproducible.

## Consequences

Positive:

- Intervals widen correctly when trials are correlated; `test_stats.py` checks the cluster
  interval against a naive flat bootstrap on highly clustered data, and BCa coverage on a known
  distribution.
- Reducing to per-sample means makes the paired tests valid for any number of trials and for
  continuous and binary metrics alike.
- Requiring both significance and a CI that excludes zero, plus Holm, keeps false regressions
  rare; a property test asserts identical configurations never produce a `regressed` verdict.
- Every step is short, readable code that can be audited against textbook definitions.

Negative:

- Per-sample means discard within-sample information that a mixed-effects model would use, so
  tests are conservative when trials are many and samples few.
- The permutation test assumes symmetric differences under H0; the bootstrap and normal-theory
  MDE are approximate for very small n (hence the warning).
- McNemar on per-sample means thresholds at 0.5, which is only meaningful for single-trial
  binary metrics, the only case in which it is selected.
- Maintaining statistical code in-house means correctness rests on our own tests rather than
  on a widely used library's.

## Alternatives considered

- **Flat (i.i.d.) bootstrap over trials.** Simpler; understates uncertainty with repeated trials.
- **Welch t-test on trial values.** Ignores pairing and clustering; poor for binary metrics.
- **`scipy.stats.bootstrap` / `permutation_test`.** Capable, but neither implements a two-stage
  cluster bootstrap, and wrapping them obscures exactly which unit is resampled.
- **Mixed-effects or Bayesian hierarchical models.** More statistically efficient, but add heavy
  dependencies and assumptions that are harder to explain in a CI failure message.
- **BCa as the default interval.** Better small-sample coverage for skewed statistics, but its
  jackknife does not extend cleanly to the two-stage cluster case; it stays available for
  single-level data.
