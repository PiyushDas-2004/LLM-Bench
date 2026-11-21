# CI eval gate

`llmbench check` compares a candidate run against a stored baseline and fails the CI job when a
gated metric regresses by more than a tolerance **and** the regression is statistically
significant. This page explains the gate, the baseline workflow built on run bundles, and how to
wire it into GitHub Actions or GitLab CI. Statistical details are in
[METHODOLOGY.md](METHODOLOGY.md) and [ADR 0004](adr/0004-bootstrap-design.md).

## How the gate decides

`tracking/gates.py::check` calls `compare_runs` restricted to the gated metrics, then evaluates
every gate on every configuration pair. Configurations are `(model_id, prompt_id)` pairs; by
default the gate compares configurations present in both runs, or the single configuration of
each run when both have exactly one. `--pair base_model/prompt=cand_model/prompt` (repeatable)
pairs them explicitly.

For each gate and pair, the comparison reduces both runs to per-sample means on their common
samples, computes a paired-bootstrap CI for the delta, runs the paired test (sign-flip
permutation by default, McNemar for single-trial binary metrics) and adjusts p-values across the
gated metrics with the configured correction. A gate **fails** when:

1. the metric moved in the bad direction by more than `max_drop` (absolute units of the metric),
   and, if `max_relative_drop` is set and the baseline value is non-zero, also by more than
   `max_relative_drop` relative to the baseline; and
2. the adjusted p-value is below `alpha` and the delta CI excludes zero.

A gate also fails when the metric is missing from either run or fewer than two samples align
("insufficient aligned data"). A drop beyond tolerance that is not significant passes, with
the reason recorded. The direction comes from the gate's `direction` or, if omitted, from the
metric's registered `higher_is_better`.

## gates.yaml

Schema (`GateConfig`, `Gate`; unknown keys are rejected):

| Key | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | float in (0, 1) | `0.05` | Significance level for every gate. |
| `correction` | `holm`, `bh`, `none` | `holm` | Multiple-comparison adjustment across gated metrics. |
| `allow_dataset_drift` | bool | `false` | Compare runs with different dataset hashes (on common sample ids). |
| `gates` | list, at least one | required | Gated metrics. |
| `gates[].metric` | string | required | Registered metric name, e.g. `pass`, `f1`, `latency_ms`. |
| `gates[].direction` | `higher_is_better`, `lower_is_better` | from registry | Override the metric's direction. |
| `gates[].max_drop` | float >= 0 | `0.0` | Absolute tolerance in the metric's units. |
| `gates[].max_relative_drop` | float >= 0 | unset | Additional relative tolerance (both must be exceeded). |
| `gates[].alpha` | float in (0, 1) | unset | Per-gate significance level. |

`examples/gates.yaml` (also scaffolded by `llmbench init`):

```yaml
alpha: 0.05
correction: holm
gates:
  - {metric: pass, max_drop: 0.02}
  - {metric: f1, max_drop: 0.03}
  - {metric: latency_ms, direction: lower_is_better, max_drop: 100}
```

Without `--gates`, `check` uses a single gate: `pass` with `max_drop: 0.02`. Every run computes
`pass`, the reliability metrics, `latency_ms`, `ttft_ms`, `tokens_per_second`, `queue_wait_ms`
and `cost_usd` (`metrics/engine.py::ALWAYS_COMPUTED`); other gated metrics must be listed in the
experiment's `metrics`.

## `llmbench check`

```bash
llmbench check --baseline main --candidate latest --gates gates.yaml --output gate.md
```

`--baseline` (default `main`) and `--candidate` (default `latest`) accept a baseline name, run
id, unique id prefix or `latest`. The command prints a table and, with `--output`, writes
Markdown: the gate results followed by the full comparison (estimates, delta CIs, tests,
adjusted p-values, effect sizes, per-sample regressions, tag breakdown).

| Exit code | Meaning |
|---|---|
| 0 | Every gate passed. |
| 1 | At least one gate failed (regression, missing metric or insufficient data). |
| 2 | Configuration or lookup error: gates file not found, run or baseline not found, ambiguous run prefix, runs with no common configuration, dataset hash mismatch without `--allow-dataset-drift`. |

A gates file that exists but fails schema validation, or is not valid YAML, is reported as a
configuration error (exit code 2), so it can never be mistaken for a regression.

## Stored-baseline workflow

CI jobs start with an empty database, so the baseline travels with the repository as a run
bundle (`tracking/bundle.py`, format `llmbench-run/1`): run metadata, manifest, every trial and
every metric value. Paths in the manifest are stored relative to the project root (the working
directory at export time) and resolved against `--base-dir` (default `.`) on import, so run
both commands from the repository root.

```bash
# once, locally or in a trusted job: produce and commit the baseline
llmbench run evals/experiment.yaml --yes --no-cache
llmbench export latest --output baselines/main.json
git add baselines/main.json

# in every CI job
llmbench import baselines/main.json --as-baseline main
llmbench run evals/experiment.yaml --yes
llmbench check --baseline main --gates evals/gates.yaml --output gate.md
```

`export` defaults to the JSON bundle format (`--format csv` or `parquet` produce flat trial
tables instead, which cannot be imported). `import` refuses a run id that already exists unless
`--replace`. The gate reads the baseline's stored trials and metric values; it does not
re-execute or re-score the baseline. The imported run keeps its original `created_at`, so the
candidate run is `latest` after it executes.

### Refreshing the baseline

Refresh after an intentional change (new prompt, new model, updated dataset) has been reviewed:

1. On the default branch, run the experiment with `--no-cache` so the baseline reflects the
   provider's current behaviour rather than cached responses.
2. `llmbench export latest --output baselines/main.json`.
3. Open a pull request with the new bundle; the diff shows changed outputs per trial. Reviewers
   can run `llmbench compare` between the old and new bundle (import both with distinct
   baseline names) to see exactly what moved.

A dataset change always requires a refresh, because the gate refuses to compare runs with
different dataset hashes (below).

## Choosing `max_drop`: tolerance versus detectability

`max_drop` says how large a regression matters; significance says whether the benchmark can
tell it from noise. Every comparison reports the minimum detectable effect (`mde`, two-sided
paired test at power 0.8, from the observed per-sample differences) and warns below 30 paired
samples. Run the experiment twice with no change (for example two runs with caching disabled)
and inspect the MDE with `llmbench compare <run1> <run2> --format md`:

- If `max_drop` is well below the MDE, regressions between the two values will often be
  non-significant and pass. The gate is then effectively set at roughly the MDE, not at
  `max_drop`. Either accept that, or lower the MDE with more samples (largest effect) or more
  trials per sample (reduces within-sample noise).
- If `max_drop` is above the MDE, small but real changes are detected statistically but
  tolerated by design. This is usually the right setting for noisy metrics such as
  `latency_ms`.
- `stats.power.required_samples(sd_diff, effect)` estimates how many samples are needed to
  detect a given effect.

## Why dataset changes fail the gate

Each run stores the content hash of the samples actually evaluated (after sampling). Different
samples have different difficulty, so a metric difference between two datasets cannot be
attributed to the system under test. `check` therefore exits 2 on a hash mismatch. Passing
`--allow-dataset-drift` (or `allow_dataset_drift: true`) compares on the intersection of sample
ids and adds a note to the report; use it deliberately, for example when samples were only
appended, and refresh the baseline afterwards.

## GitHub Actions example (another repository)

The repository under test commits `evals/experiment.yaml`, `evals/gates.yaml`, the dataset and
prompts it references, and `baselines/main.json`. Pin LLMBench to a commit so the gate's
statistics do not change underneath you.

```yaml
name: LLM eval gate
on:
  pull_request:
permissions:
  contents: read
jobs:
  eval-gate:
    runs-on: ubuntu-latest
    env:
      LLMBENCH_DB_URL: sqlite:///${{ github.workspace }}/.llmbench/ci.db
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install LLMBench
        # or, for a vendored copy: pip install "./tools/llmbench[openai]"
        run: pip install "llmbench[openai] @ git+https://github.com/PiyushDas-2004/LLMBench@<commit-sha>"
      - name: Import stored baseline
        run: llmbench import baselines/main.json --as-baseline main
      - name: Run candidate experiment
        run: llmbench run evals/experiment.yaml --yes --strict-models
      - name: Eval gate
        run: llmbench check --baseline main --gates evals/gates.yaml --output gate.md
      - name: Publish gate report
        if: always()
        run: '[ -f gate.md ] && cat gate.md >> "$GITHUB_STEP_SUMMARY" || true'
```

Notes:

- `llmbench run` exits 2 if the cost estimate exceeds `cost_confirm_threshold_usd` without
  `--yes`, and 3 if the run ended `budget_exceeded` or `partial`; either stops the job before
  the gate, which is intended.
- `--strict-models` turns a missing key or placeholder model name into an error instead of a
  silently skipped model.

### Real providers and secrets

- Store keys as repository or environment secrets (`OPENAI_API_KEY`, `GEMINI_API_KEY`, or the
  variable named by a model's `api_key_env`). Keys are read only from the environment.
- Pull requests from forks do not receive secrets; restrict the job to same-repository pull
  requests or run the mock-provider variant for forks.
- Set `budget.max_usd` and `budget.max_requests` in the experiment; `llmbench plan` prints an
  upper-bound estimate (it needs a `pricing.yaml` entry for each model).
- Pin dated model snapshots in the experiment YAML where the provider offers them; an alias
  that moves will register as a regression or improvement unrelated to your change.
- Set `rate_limit` to your account limits. Limits apply per provider endpoint (provider,
  `base_url`, `api_key_env`), shared by all models that use it.

## GitLab CI

```yaml
eval-gate:
  image: python:3.12-slim
  variables:
    LLMBENCH_DB_URL: "sqlite:///$CI_PROJECT_DIR/.llmbench/ci.db"
  script:
    - pip install "llmbench[openai] @ git+https://github.com/PiyushDas-2004/LLMBench@<commit-sha>"
    - llmbench import baselines/main.json --as-baseline main
    - llmbench run evals/experiment.yaml --yes --strict-models
    - llmbench check --baseline main --gates evals/gates.yaml --output gate.md
  artifacts:
    when: always
    paths: [gate.md]
```

`python:3.12-slim` has no `git`; install it first or use a full Python image. Define
`OPENAI_API_KEY` as a masked CI/CD variable.

## Dogfooding: this repository's gate

`.github/workflows/eval-gate.yml` gates LLMBench with LLMBench, fully offline on the mock
provider:

1. imports `baselines/gate-demo.json` (a bundle of `examples/gate/baseline.yaml`) as `main`;
2. runs `examples/gate/noise.yaml`, which changes only `mock_seed` (sampling noise), and
   requires `llmbench check --baseline main --gates examples/gates.yaml` to pass;
3. runs `examples/gate/regression.yaml`, which lowers the mock `accuracy` from 0.85 to 0.40,
   and requires the same check to exit 1;
4. appends both Markdown reports to the job summary.

Mock-provider numbers are illustrative and say nothing about any real model. The gate Markdown
does not carry the mock banner that the HTML and run reports show, so label such summaries in
the workflow when publishing them.
