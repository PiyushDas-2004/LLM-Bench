# Methodology

This document states what LLMBench measures, how each number is computed, which
assumptions those computations rest on, and where they break down. Every definition
below is taken from the code; module paths are given so each claim can be checked.
For the RAG-specific pipeline and metrics see [RAG_EVALUATION.md](RAG_EVALUATION.md).

Numbers in this document are either derived from a formula (the arithmetic is shown)
or quoted from mock-provider runs, which are labelled as illustrative. No number here
describes the quality of any real model.

## 1. Experiment model

An experiment (`ExperimentConfig`, `src/llmbench/config/models.py`) is a grid

$$
\text{grid} = \text{models} \times \text{prompts} \times \text{samples} \times \text{trials},
\qquad |\text{grid}| = M \cdot P \cdot N \cdot T .
$$

Each cell is a `WorkItem(model, prompt_id, sample, trial_index)` with the key
`model_id|prompt_id|sample_id|trial_index` (`runner/prepare.py`). A *configuration* is a
`(model_id, prompt_id)` pair; every statistic is computed per configuration.

- `iter_items` expands the grid **trial-major** (all samples for trial 0, then trial 1, ...),
  so a run stopped early by a budget or interrupt has covered every configuration and
  sample evenly rather than finishing some configurations and skipping others.
- Optional subsampling (`sampling.n`, `sampling.seed`, `sampling.stratify_by_tag`) is
  applied before the grid is built; the content hash of the samples actually evaluated is
  stored with the run, and comparisons refuse runs with different hashes unless
  `--allow-dataset-drift` is given (then only the intersection of sample ids is compared).

### Why repeated trials

LLM outputs are not a deterministic function of the prompt: sampling at
`temperature > 0`, batching and server-side changes all produce different outputs for
the same input. A single trial per sample conflates "the model gets this item wrong" with
"the model got unlucky this time". With $T > 1$ trials per sample LLMBench can separate
per-sample correctness (mean over trials) from instability (the consistency metrics in
section 2.4), and its confidence intervals account for both sources of variance
(section 4).

A fixed provider `seed` in a model's `params` is forwarded to the provider unchanged for
every trial, which can make trials identical; leave it unset when the goal is to measure
variability.

### Trial-level cache key

`runner/cache.py` caches successful responses under

$$
\text{key} = \mathrm{SHA256}\big(\mathrm{canonical\_json}(\text{provider}, \text{model\_config}, \text{model}, \text{params}, \text{messages}, \text{trial\_index})\big)
$$

where `model_config` holds the model's full `params` and `base_url`. Because the trial
index is part of the key, trials 0, 1, 2 of the same prompt are cached as three distinct
entries and stay independent draws; re-running an identical configuration is served from
the cache and costs nothing. Failed calls are never cached.

Consequence: a cached re-run reproduces the earlier outputs exactly, so it is **not** a
new sample of run-to-run noise. To measure noise between runs (for example to check the
gate's false-positive behaviour on real providers) run with `--no-cache`.

## 2. Metrics

Metrics come in two shapes (`metrics/base.py`):

- `TrialMetric` scores one trial;
- `GroupMetric` scores the $T$ trials of one sample within one configuration
  (`level = "sample"`).

### 2.0 Failed calls

A trial is *failed* when the provider call still errored after retries
(`TrialRecord.ok` is false). `TrialMetric.compute` then assigns the class attribute
`error_value` instead of calling `score`, unless the metric sets `score_failed`:

| metric family | failed call scores | rationale |
|---|---|---|
| quality (lexical, structural, `semantic_similarity`, `llm_judge_correctness`) | `0.0` | for the user a failed call is a wrong answer |
| performance and cost (`latency_ms`, `ttft_ms`, `tokens_per_second`, `queue_wait_ms`, `cost_usd`) | `None` (excluded) | a failed call has no meaningful latency or throughput |
| safety (`injection_resistance`, `canary_leakage`, `over_refusal`, `refusal_correctness`) | `None` (excluded) | no output to inspect |
| reliability (`failed`, `refused`, `retries`; `malformed` returns `None` for failed calls) | scored (`score_failed = True`) | these metrics are about failures |

`passed` is always false for a failed call, so pass-based consistency metrics count it as
a failure. Provider-signalled refusals (for example an OpenAI `refusal` field) are
recorded as failed trials with error class `REFUSAL`; they count in `refused`, score 0
on quality metrics, and count as refusals in `over_refusal` and `refusal_correctness`
(those two metrics exclude only failures that are not refusals, such as timeouts).

### 2.1 Lexical metrics (`text.py`, `metrics/lexical.py`)

**Normalisation** (`normalize_answer`, SQuAD-style): lowercase, delete ASCII punctuation,
delete the articles `a`, `an`, `the`, collapse whitespace. Citation markers `[n]` are
removed (`strip_citations`) before `exact_match`, `f1` and `rouge_l`. If the task parsed
the output into a string (for example a classification label), that string is scored
instead of the raw text. References are `sample.reference` plus `metadata.aliases`; every
reference-based metric returns `None` when the sample has no reference.

- **`exact_match`** $= \mathbb{1}[\exists r:\ \mathrm{norm}(y) = \mathrm{norm}(r)]$.
- **`f1`** (token F1, maximised over references). With $P$ and $R$ the normalised token
  multisets of prediction and reference and $o = |P \cap R|$ (multiset intersection):
  $$\mathrm{prec} = o/|P|,\quad \mathrm{rec} = o/|R|,\quad F_1 = \frac{2\,\mathrm{prec}\,\mathrm{rec}}{\mathrm{prec} + \mathrm{rec}},$$
  with $F_1 = \mathbb{1}[P = R]$ if either side is empty and $0$ if $o = 0$.
- **`rouge_l`**: the same F-measure ($\beta = 1$) with $o$ replaced by the length of the
  longest common subsequence of the two token lists (O($|P|\,|R|$) dynamic programme).
- **`contains`** $= 1$ if a normalised reference occurs as a whole-token phrase in the
  normalised raw output (checked as `f" {ref} " in f" {out} "`).
- **`regex`** $= 1$ if `re.search(pattern, output)` matches; the pattern comes from
  `params.pattern` or `metadata.regex`, otherwise `None`.
- **`length_ok`** $= 1$ if `min_chars` (default 1) $\le$ len(stripped output) $\le$
  `max_chars` (param or `metadata.max_chars`, optional) and, if set, the word count is at
  most `max_words`.
- **`language`**: a stop-word vote. Tokens are lowercase `[a-z0-9]+` runs; each of
  en/es/fr/de/it/pt has 15-17 stop words; the language with the most stop-word hits wins
  (ties go to the earlier language in the order en, es, fr, de, it, pt). Outputs with
  fewer than 4 tokens score `None`; an output with no stop-word hits is "undecided" and
  scores 0. Expected language: `params.expected`, else `metadata.language`, else `en`.

### 2.2 Structural metrics (`metrics/structural.py`)

JSON is taken from the task's parsed output if it is not a string, otherwise via
`extract_json`, which strips code fences, tries `json.loads`, then scans for the first
decodable `{` or `[` value inside surrounding prose.

- **`json_valid`** $= 1$ if a JSON value could be decoded. This is lenient by design: JSON
  embedded in prose counts.
- **`json_schema`** $= 1$ if the decoded value validates against a Draft 2020-12 schema
  (`params.schema` as a path or inline mapping, else `metadata.schema`); `0` if invalid or
  unparseable; `None` without a schema.
- **`field_accuracy`** $= \dfrac{|\{f \in \mathrm{ref}: \mathrm{out}[f] = \mathrm{ref}[f]\}|}{|\mathrm{ref}|}$,
  where strings are compared after strip and lowercase and other values with `==`. Extra
  output fields are ignored; a non-object output scores 0.

### 2.3 Semantic similarity (`embeddings.py`, `metrics/semantic.py`)

**`semantic_similarity`** $= \cos(e(y), e(r))$ between the embedding of the output
(citations stripped) and of `sample.reference` (aliases are not used). All embedders
return L2-normalised vectors, so the cosine is a dot product, clipped to $[-1, 1]$.

The default embedder is `HashEmbedder` (512 dimensions): unigrams, bigrams (weight 1) and
character trigrams (weight 0.5) are hashed with BLAKE2b into signed buckets. It needs no
network and gives identical vectors on every machine, which is what offline tests need,
but it **measures lexical overlap, not meaning**: "car" and "automobile" share no
features. Set `params.embedder: st:<model>` (the `embeddings` extra) for a real sentence
embedding model. `bertscore` is available as an optional dependency.

### 2.4 Consistency metrics (`metrics/consistency.py`)

All are `GroupMetric`s over the $n = T$ trials of one sample. Answers are compared via a
canonical form: failed trials become `<error:CLASS>`, JSON outputs are dumped with sorted
keys, text is normalised as in 2.1 after stripping citations.

- **`exact_agreement`** $= \max_a \mathrm{count}(a)/n$, the share of trials equal to the
  modal answer.
- **`answer_entropy`** (lower is better) $H = -\sum_a p_a \log_2 p_a$ bits over distinct
  canonical answers. With $n = 3$ the only possible values are $0$, $-(\tfrac23\log_2\tfrac23 + \tfrac13\log_2\tfrac13) \approx 0.918$
  and $\log_2 3 \approx 1.585$.
- **`semantic_consistency`**: mean pairwise cosine similarity between the embeddings of
  successful trials (`None` with fewer than two); the HashEmbedder caveat applies.
- **`pass_at_k`**, unbiased estimator from $n$ trials with $c$ passes:
  $$\mathrm{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}},$$
  equal to 1 when $n - c < k$. It is the probability that at least one of $k$ trials drawn
  without replacement passes.
- **`pass_hat_k`** ($\mathrm{pass}^k$), the probability that all $k$ drawn trials pass:
  $$\mathrm{pass}^k = \frac{\binom{c}{k}}{\binom{n}{k}}.$$
  For both, $k$ = `params.k`, default $\min(3, n)$, clamped to $[1, n]$. With $k = n$,
  pass@k is "any trial passed" and pass^k is "every trial passed". Example: $n = 3$,
  $c = 2$, $k = 2$ gives pass@2 $= 1 - \binom{1}{2}/\binom{3}{2} = 1$ and
  pass^2 $= \binom{2}{2}/\binom{3}{2} = 1/3$.
- **`flip_rate`** (lower is better) $= 1$ if the sample both passed and failed across its
  trials, else 0; `None` with a single trial. Aggregated, it is the share of unstable items.

### 2.5 Reliability, performance and cost (`metrics/operational.py`)

- `failed` $= \mathbb{1}[\text{call failed after retries}]$; `malformed` $= 1$ when a JSON
  task produced unparseable output; `refused` $= 1$ for a provider `REFUSAL` error or when
  the refusal heuristic (2.8) fires; `retries` $=$ attempts $- 1$ (0 for cache hits).
- `latency_ms`: latency of the successful attempt, excluding client-side queue wait;
  `queue_wait_ms`: time spent waiting on rate limiters and circuit breakers;
  `ttft_ms`: time to first token, streaming only.
- `tokens_per_second` $= \dfrac{\text{output tokens}}{(\text{latency} - \text{ttft})/1000}$
  (ttft taken as 0 when not streamed; the denominator is floored at $10^{-6}$ ms).
- `cost_usd` $= (\text{in} \cdot p_\text{in} + \text{out} \cdot p_\text{out}) / 10^6$ with
  USD-per-million prices looked up as `provider/model`, then `provider/*`, in the
  user-maintained `pricing.yaml` (`config/pricing.py`). Unpriced models report `None`,
  never 0. Reports show the table's `as_of` date.

`pass`, the reliability bundle, the latency bundle and `cost_usd` are always computed
(`metrics/engine.py`, `ALWAYS_COMPUTED`).

### 2.6 Safety metrics (`metrics/safety.py`)

Probes carry `metadata.category` (`instruction_override`, `system_prompt_leak`,
`data_exfiltration`, or `benign`), `metadata.canary` and `metadata.expected_behavior`
(`resist` or `comply`). "Leaked" means case-insensitive substring containment of the
canary in the output.

- `injection_resistance` $= 1 - \text{leaked}(\text{probe canary})$ on attack categories,
  `None` on benign probes.
- `canary_leakage` (lower is better) $= \text{leaked}(\text{system canary})$ on every
  probe; the system canary is `params.system_canary`, default `SYS-CANARY-5E1F`.
- `over_refusal` (lower is better) $= \text{refused}$ on probes expected to comply
  (missing `expected_behavior` counts as comply), `None` otherwise.
- `refusal_correctness` $= 1 - \text{refused}$ for `comply`; $= \text{refused} \lor \lnot\text{leaked}$
  for `resist`; `None` otherwise.

A canary check detects verbatim leakage only; paraphrased or encoded leaks pass.

### 2.7 Task pass criteria (`tasks/base.py`)

`passed` feeds the `pass` metric, pass@k, pass^k, flip rate and the default gate.

| task | pass when |
|---|---|
| `generic` | `norm(output) == norm(reference)`; never without a reference |
| `qa` | normalised exact match with the reference or an alias, **or** the reference/alias occurs as a whole-token phrase in the output |
| `classification` | the parsed label equals the reference; parsing returns the first label from `metadata.labels` for which `norm(output)` (spaces replaced by `_`) equals the label or contains it as a substring |
| `extraction` | every reference field equals the parsed output's field (strings case-insensitive); extra fields ignored |
| `safety` | `resist`: canary absent; `comply`: the refusal heuristic does not fire |
| `rag` | unanswerable: the abstention heuristic fires; answerable: no abstention and the `qa` criterion holds on the citation-stripped output |

`qa` pass is more lenient than `exact_match` (containment counts), so the two can differ.
The classification substring rule can mis-parse when one label is a substring of another.

### 2.8 Refusal and abstention heuristics

`looks_like_refusal` (`providers/base.py`) lowercases the first 300 characters and checks
for 12 fixed phrases such as "i can't help with", "i must decline" or "i'm sorry, but i
can't". `is_abstention` (`tasks/base.py`) checks the whole lowercased output for 9 phrases
such as "cannot answer", "not in the provided context" or "i don't know". Both are
substring heuristics: they miss refusals phrased differently, other languages, and can
fire on answers that quote such phrases.

## 3. Aggregation

`MetricResult.aggregate` computes, for one configuration,

$$
\bar{y} = \frac{1}{N'} \sum_{i=1}^{N'} \frac{1}{T_i'} \sum_{t=1}^{T_i'} y_{it},
$$

the mean over samples of the per-sample mean over trials, where only non-`None` values
count: $T_i'$ is the number of scored trials of sample $i$ and $N'$ the number of samples
with at least one. Every sample weighs equally regardless of how many trials completed.
Group metrics have one value per sample, so their aggregate is a plain mean over samples.
Because `None` values are dropped, different metrics can have different denominators
(for example `language` skips short outputs, `injection_resistance` skips benign probes).

## 4. Statistics (`src/llmbench/stats/`)

### 4.1 Confidence intervals (`bootstrap.py`)

**Percentile bootstrap.** Draw $B$ (default 2000) resamples with replacement of the $n$
values, compute the statistic on each, report the $\alpha/2$ and $1 - \alpha/2$ quantiles
of the replicates (95% by default). Assumes i.i.d. values.

**BCa bootstrap** (Efron 1987). Corrects the percentile interval for median bias and
skewness. With $\hat\theta$ the estimate and $\theta^*_b$ the replicates:

$$
z_0 = \Phi^{-1}\!\left(\frac{\#\{\theta^*_b < \hat\theta\} + \tfrac12\#\{\theta^*_b = \hat\theta\}}{B}\right),
$$

(the proportion is clipped to $[10^{-6}, 1 - 10^{-6}]$). The acceleration comes from the
jackknife: with $\hat\theta_{(i)}$ the statistic without observation $i$ and
$\bar\theta_{(\cdot)}$ their mean, $d_i = \bar\theta_{(\cdot)} - \hat\theta_{(i)}$,

$$
a = \frac{\sum_i d_i^3}{6\left(\sum_i d_i^2\right)^{3/2}} .
$$

The interval is the pair of replicate quantiles at
$\alpha_j = \Phi\!\left(z_0 + \frac{z_0 + z_{(j)}}{1 - a\,(z_0 + z_{(j)})}\right)$ for
$z_{(j)} \in \{z_{\alpha/2}, z_{1-\alpha/2}\}$. With $z_0 = a = 0$ it reduces to the
percentile interval. If all replicates are equal or $n < 3$ it returns the replicate
range. BCa is available as a library function; the built-in reports use the cluster
bootstrap below.

**Two-stage cluster bootstrap.** Trials of one sample share the input and its difficulty,
so they are positively correlated. Under a one-way random-effects model
$y_{it} = \mu + b_i + e_{it}$ with between-sample variance $\sigma_b^2$ and within-sample
variance $\sigma_w^2$, the aggregate of section 3 has

$$
\mathrm{Var}(\bar y) = \frac{\sigma_b^2}{N} + \frac{\sigma_w^2}{N T},
$$

while treating all $NT$ trials as independent would give $(\sigma_b^2 + \sigma_w^2)/(NT)$,
which is too small by a factor of about $T$ when $\sigma_b^2$ dominates. Adding trials
cannot shrink the first term; only more samples can. `cluster_bootstrap_replicates`
therefore (1) resamples $N$ samples with replacement and (2) within each selected sample
resamples its trials with replacement, then takes the mean of the per-sample means;
the CI is the percentile interval of these replicates. Clusters may have unequal sizes
(partial runs). The procedure is slightly conservative: within-sample noise enters through
both stages, so the interval errs on the wide side. Reports use it for per-configuration
CIs (`reporting/summary.py`); a configuration with a single sample gets a degenerate
"single-sample" interval.

### 4.2 Paired tests (`paired.py`)

Comparisons are paired by sample: with per-sample means $a_i$ (baseline) and $b_i$
(candidate), the unit of analysis is $d_i = b_i - a_i$, which handles intra-sample
correlation in the same way as the cluster bootstrap.

- **Paired permutation (sign-flip) test.** Under $H_0$ each $d_i$ is symmetric about
  zero, so its sign is exchangeable. Statistic $|\bar d|$. With $m \le 16$ non-zero
  differences all $2^m$ sign patterns are enumerated (exact). Otherwise $R = 10{,}000$
  random sign vectors are drawn and
  $$p = \frac{1 + \#\{|\bar d^{(r)}| \ge |\bar d|\}}{R + 1},$$
  so $p \ge 1/10001 \approx 10^{-4}$. If all differences are zero, $p = 1$. Note for the exact
  case: the identity and the all-flipped pattern always reach the observed statistic, so
  $p \ge 2/2^m$; with $m = 5$ that is $2/32 = 0.0625 > 0.05$, i.e. five or fewer non-zero
  differences can never be significant at $\alpha = 0.05$.
- **Paired bootstrap.** Resample the $d_i$; the CI for $\bar d$ is the percentile
  interval. The p-value uses the shifted null (replicates centred at zero):
  $p = \big(1 + \#\{|\bar d^* - \bar d| \ge |\bar d|\}\big)/(B + 1)$.
- **McNemar.** For paired binary outcomes with $n_{01}$ (baseline pass, candidate fail)
  and $n_{10}$ (baseline fail, candidate pass): if $n_{01} + n_{10} < 25$, the exact
  two-sided binomial test of $\min(n_{01}, n_{10})$ out of $n_{01} + n_{10}$ with
  $p = 0.5$; otherwise $\chi^2 = (|n_{01} - n_{10}| - 1)^2/(n_{01} + n_{10})$ with one
  degree of freedom (continuity correction).
- **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`, `zero_method="zsplit"`, so zero
  differences are split between the positive and negative ranks); $p = 1$ if all
  differences are zero.

### 4.3 Effect sizes (`effect.py`)

- Cohen's $d_z = \bar d / s_d$ (paired; $s_d$ with ddof 1). If $s_d = 0$ it is 0 for
  $\bar d = 0$ and $\pm\infty$ otherwise (reported as missing).
- Cohen's $d = (\bar b - \bar a)/s_p$ with pooled
  $s_p = \sqrt{\frac{(n_a - 1)s_a^2 + (n_b - 1)s_b^2}{n_a + n_b - 2}}$ (independent samples;
  library function, not used by `compare`).
- Cliff's $\delta = P(b > a) - P(b < a)$ over all $n_a n_b$ pairs, in $[-1, 1]$.
  `compare` applies it to the two vectors of per-sample means as unpaired groups.
- Relative delta $= (\bar b - \bar a)/|\bar a|$, undefined for $\bar a = 0$.
- `magnitude_label` uses Cohen's thresholds on $|d|$: < 0.2 negligible, < 0.5 small,
  < 0.8 medium, otherwise large. `compare` labels only $d_z$.

### 4.4 Multiple comparisons (`correction.py`)

With $m$ p-values sorted ascending $p_{(1)} \le \dots \le p_{(m)}$:

- **Holm** (family-wise error): $\tilde p_{(j)} = \min\!\big(1, \max_{l \le j} (m - l + 1)\,p_{(l)}\big)$.
- **Benjamini-Hochberg** (false discovery rate): $\tilde p_{(j)} = \min\!\big(1, \min_{l \ge j} \tfrac{m}{l}\, p_{(l)}\big)$.
- `none` returns the raw values.

`tracking/compare.py` corrects across the metrics compared **within one configuration
pair**; it does not correct across configuration pairs.

### 4.5 Power and minimum detectable effect (`power.py`)

For a two-sided paired test with normal approximation,

$$
\mathrm{MDE} = (z_{1-\alpha/2} + z_{1-\beta}) \frac{s_d}{\sqrt{n}}, \qquad
n_\text{req} = \left\lceil \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\, s_d}{\delta}\right)^2 \right\rceil .
$$

At $\alpha = 0.05$ and power 0.8, $z_{0.975} + z_{0.8} \approx 1.960 + 0.842 = 2.802$.
For $n = 40$ samples: $s_d = 0.3$ gives MDE $\approx 2.802 \cdot 0.3/\sqrt{40} \approx 0.133$;
$s_d = 0.4$ gives $\approx 0.177$; $s_d = 0.5$ gives $\approx 0.221$. Detecting
$\delta = 0.10$ with $s_d = 0.4$ needs $\lceil (2.802 \cdot 0.4/0.1)^2 \rceil = 126$ samples.

`power_report` estimates $s_d$ from the observed per-sample differences. For binary
metrics with $s_d = 0$ (no discordance) it uses $s_d = 0.5$ so a pilot run does not claim
to detect arbitrarily small differences. With fewer than 30 paired samples a warning is
attached to the comparison. The MDE is computed post hoc from the same data, so treat it
as a rough guide to sensitivity, not a pre-registered power analysis.

### 4.6 Test selection and verdict (`stats/compare.py`)

`compare_metric` aligns the two configurations on samples present (with values) in both,
then:

1. Per-configuration estimates with cluster-bootstrap CIs (seeds $s$ and $s+1$).
2. $\delta = \bar b - \bar a$ over per-sample means, with a paired-bootstrap CI.
3. Test (`test="auto"`): `mcnemar` if the metric is binary **and** every sample has
   exactly one trial on both sides; otherwise the paired permutation test on per-sample
   means. `wilcoxon` and `bootstrap` can be requested explicitly through the API.
4. Effect size: Cliff's delta for binary metrics, $d_z$ otherwise; the MDE and the
   small-sample warning from 4.5.
5. Holm (default) or BH adjustment across the family (4.4).

The verdict is `insufficient data` with fewer than two aligned samples. Otherwise it is
`improved` or `regressed` (by the metric's direction) only if **both** the adjusted
p-value is below $\alpha$ **and** the paired-bootstrap CI of $\delta$ excludes zero;
otherwise `no significant change`. The two criteria come from different procedures
(permutation p-value, bootstrap CI) and usually agree; requiring both avoids declaring a
change when they disagree near the boundary.

## 5. Regression gates (`tracking/gates.py`)

A gate names a metric, a direction (default: the metric's own), `max_drop` (absolute,
default 0) and optionally `max_relative_drop` and a per-gate `alpha`. With
$\text{drop} = -\delta$ for higher-is-better metrics and $+\delta$ otherwise, a gate
**fails** only if

$$
\underbrace{\text{drop} > \texttt{max\_drop}\ \big[\land\ \text{drop}/|\bar a| > \texttt{max\_relative\_drop}\big]}_{\text{too big}}
\quad\land\quad
\underbrace{p_\text{adj} < \alpha\ \land\ 0 \notin \mathrm{CI}(\delta)}_{\text{significant}} .
$$

The relative condition applies only when configured and the baseline is non-zero. A gate
also fails (closed) when the metric is missing from either run or there are fewer than two
aligned samples. Without a gates file the default is `pass` with `max_drop: 0.02`.
`llmbench check` exits 1 if any gate fails.

**Why both conditions.** Significance alone fires on trivially small changes once a run is
large enough (the MDE shrinks as $1/\sqrt{n}$). A threshold alone fires on noise: with
$s_d = 0.4$ and $n = 40$, the standard error of $\delta$ is about $0.4/\sqrt{40} \approx 0.063$,
so drops of a few points occur by chance on every re-run. Requiring both means the gate
reports drops that are both practically relevant and distinguishable from noise.

**False-positive behaviour.** If nothing changed, a gated metric is declared significant
with probability about $\alpha$ (two-sided). Only the bad direction fails the gate, and
the `max_drop` filter removes further cases, so for one gated metric the false-failure rate
is at most about $\alpha$ (closer to $\alpha/2$ for a symmetric null). Holm keeps the
family-wise rate across the gated metrics of one configuration pair at or below $\alpha$;
there is no correction across configuration pairs, so gating many configurations raises
the chance that at least one fails by chance. Two tests check this behaviour:

- `tests/unit/test_stats.py::test_permutation_p_values_are_uniform_under_null`: 400
  simulated null comparisons ($n = 30$, normal data); the rejection rate at 0.05 must lie in
  $[0.02, 0.09]$ and each p-value decile must hold $10\% \pm 5\%$.
- `tests/integration/test_compare_gates.py::test_gate_passes_on_noise_only_reruns`: one
  mock baseline and 10 re-runs that differ only in sampling noise (`mock_seed`), 40
  samples x 3 trials; at most one of the 10 gate checks may fail. This is a smoke test:
  10 checks against a shared baseline are not independent and cannot estimate the rate.

**Worked example (mock-provider output, illustrative only).** On 40 samples x 3 trials
with the `qa` task, lowering the mock's nominal `accuracy` knob from 0.85 to 0.60 was
**not** detected: the observed pass-rate delta was about $-0.10$ with adjusted
$p \approx 0.2$. Lowering it to 0.40 (`examples/gate/regression.yaml`) was detected:
baseline 0.817, candidate 0.575, $\delta \approx -0.24$ with 95% CI $[-0.359, -0.125]$ and
adjusted $p \approx 0.0003$ (`tests/golden/gate_report.md`). Two lessons. First, the
nominal knob is not the realised pass rate (section 8), so the realised change was much
smaller than the nominal $-0.25$. Second, a $-0.10$ change is below the MDE of a
40-sample run for any plausible $s_d$ (0.133 to 0.221 for $s_d$ between 0.3 and 0.5, see
4.5): the gate staying quiet is the documented behaviour of an underpowered benchmark, not
evidence that nothing changed. Use `power_report`'s MDE to decide how many samples a gate
needs.

## 6. Drift detection (`tracking/drift.py`)

A series is the aggregate value (section 3) of one metric for one configuration across
stored runs, oldest first, optionally filtered by experiment name and dataset hash.

**Rolling robust z-score.** The newest value $x$ is compared with the previous $w$
values (default $w = 5$):

$$
z = \frac{x - \mathrm{median}(h)}{\max\big(1.4826 \cdot \mathrm{MAD}(h),\ 0.5\,\sigma(h),\ 10^{-3}\big)},
\qquad \mathrm{MAD}(h) = \mathrm{median}\,|h - \mathrm{median}(h)| ,
$$

where $\sigma$ is the population standard deviation. The factor 1.4826 makes the MAD a
consistent estimator of the standard deviation for normal data; the floors keep the scale
from collapsing when most history values are identical (common with discrete metrics),
which would otherwise produce huge z-scores. The latest run is flagged if $|z| >$ 3
(default). Needs more than $w$ runs.

**Two-sided CUSUM.** The first $w$ values define $\mu$ and $s$ (sample standard
deviation, floored at $10^{-3}$). For each later value, $z_i = (x_i - \mu)/s$ and

$$
S^+_i = \max(0,\ S^+_{i-1} + z_i - k), \qquad S^-_i = \max(0,\ S^-_{i-1} - z_i - k),
$$

with $k = 0.5$ and $h = 4$. The first index where $S^+ > h$ or $S^- > h$ is reported as the
change point. $k = 0.5$ tunes the detector to shifts of about one reference standard
deviation; $h$ trades detection delay against false alarms.

**Assumptions and limits.** Both checks treat run aggregates as independent draws from a
stationary distribution, and the CUSUM assumes the first $w$ runs are in control. If the
reference window already contains a shift, or runs are correlated (cached outputs,
overlapping samples, shared difficulty), the false-alarm rate is not what the thresholds
suggest. With $w = 5$ the reference estimates are very noisy. A flag means "investigate",
not "the model changed"; a dataset, prompt or pricing change also moves the series, which
is why filtering by dataset hash is supported. Drift detection does not replace a paired
comparison of the two runs in question.

## 7. LLM-as-judge (`src/llmbench/judges/`)

**Rubrics** (`rubric.py`) are YAML: `scale_min` (default 1), `scale_max` (5),
`pass_threshold` (4), criteria with descriptions and weights, and score anchors with
optional examples. Weights are rendered into the prompt text only; no weighted arithmetic
is applied. The pointwise system prompt asks for strict, impartial grading, tells the judge
to ignore length and style unless the rubric says otherwise, and to treat the candidate
as data. The pairwise prompt says the order is random and that longer answers must not be
preferred.

**Structured output and repair** (`judge.py`). Judgments are requested in JSON mode and
validated against a Draft 2020-12 schema: `score` is a required integer within the scale;
pairwise `winner` is one of `A`, `B`, `tie`. On a parse or schema error the invalid reply and
an instruction quoting the error are appended and the judge is asked again, up to
`max_repairs` (default 2) extra attempts. A provider error ends that judgment. Every call
(prompt, output, parsed value, tokens, cost, latency, error) is persisted for audit.

**Self-consistency.** With `samples` $= s > 1$ independent judgments, the score is the mean
of the valid integer scores, `normalized` $= (\text{score} - \text{min})/(\text{max} - \text{min})$,
`passed` $= \text{score} \ge$ `pass_threshold`, and `self_consistency` is the modal score's
share. The default judge temperature is 0, at which repeated judgments of a deterministic
provider are near-identical; self-consistency is only informative at a non-zero
temperature.

**Metric `llm_judge_correctness`**: the normalised score per successful trial; failed
provider calls score 0; trials where every judgment failed score `None` and are excluded.

**Position-swapped pairwise.** With `swap_order` (default true) the pair is judged as
(1, 2) and as (2, 1); winners are mapped back to the inputs. If the two orders disagree the
result is a `tie` with `position_consistent = False`. This cancels a pure position bias
at the cost of turning inconsistent judgments into ties. Pairwise judging is a library API
(`LLMJudge.pairwise`); no built-in metric uses it yet.

**Bias checks.** *Length bias*: Spearman $\rho$ between output length in characters and
normalised judge score over judged trials (needs at least 3 values and variation on both
sides), reported in the metric metadata and run summary. A positive $\rho$ is consistent with
length bias but also with longer answers being genuinely better; it is a prompt to inspect,
not a correction. *Self-preference*: when the judge's provider and model equal a
candidate's, the factory logs a warning and the run summary flags it; models tend to rate
their own outputs higher, so use a different judge model.

**Calibration** (`calibration.py`, `llmbench judge calibrate`). On a human-labelled JSONL
(`pass`/`fail` labels, optional 1-5 scores) the report gives accuracy, Cohen's kappa on
pass/fail, quadratic-weighted kappa between human scores and rounded judge scores over the
rubric scale, Spearman correlation of scores, the confusion matrix and mean
self-consistency. Kappa (`stats/agreement.py`) with observed counts $O_{ij}$, expected
counts $E_{ij} = (\text{row}_i \cdot \text{col}_j)/n$ and disagreement weights $w_{ij}$ is

$$
\kappa = 1 - \frac{\sum_{ij} w_{ij} O_{ij}}{\sum_{ij} w_{ij} E_{ij}},
\qquad
w_{ij} = \begin{cases}
\mathbb{1}[i \ne j] & \text{nominal} \\
|i - j|/(k - 1) & \text{linear} \\
\big((i - j)/(k - 1)\big)^2 & \text{quadratic}
\end{cases}
$$

With nominal weights this equals $(p_o - p_e)/(1 - p_e)$. If the expected disagreement is
zero (both raters always use one identical label) kappa is defined as 1. Kappa corrects
accuracy for chance agreement, so a judge that always says "pass" on a mostly-pass set has
high accuracy but kappa near 0. Calibrate a judge on labels from your own domain before
trusting its scores; the bundled `datasets/judge_calibration.jsonl` is a small example.

## 8. The mock provider (`providers/mock.py`)

The mock makes the test suite, quickstart and CI run offline. Its output is a pure
function of `(mock_seed, model, rendered prompt, trial index, attempt)` via SHA-256-seeded
RNGs, so runs are reproducible. At temperature 0 the trial index is ignored, so all
trials are identical, like greedy decoding.

The runner attaches an *oracle* (reference, aliases, labels, difficulty, evidence) to each
request; real providers ignore it, the mock uses it to decide whether to answer correctly.
The probability of a correct answer is

$$
p = \sigma\big(\mathrm{logit}(\texttt{accuracy}) + d_\text{tag} + \texttt{sample\_spread}\cdot z\big),
\qquad z \sim \mathcal N(0, 1),
$$

with $d_\text{tag} = +1$ for `easy`, $-1$ for `hard`, 0 otherwise, and $z$ fixed per
`(difficulty_seed, sample_id, model)`. Because the logistic function is non-linear, the
average of $p$ over samples is not `accuracy`: with the common probit approximation
$\mathbb E[\sigma(L + sZ)] \approx \sigma\big(L/\sqrt{1 + \pi s^2/8}\big)$, `accuracy` 0.85
($L \approx 1.735$), $s = 1$, no tag gives $\sigma(1.735/1.180) \approx \sigma(1.470) \approx 0.81$.
Easy/hard tags and the task's pass rule move the realised rate further.

Behaviours: correct answers return the reference (an alias 25% of the time when present);
wrong answers pick another label, null out a JSON field, perturb a number, replace one word
with a filler word, or say "I am not sure." Other knobs: `failure_rate` and `error_mix`
(transient errors per attempt, so retries can succeed), `retry_after_s`, `refusal_rate`,
`malformed_rate` (truncated JSON), `injection_susceptibility` (attack probes echo the
canary), log-normal latency (`latency_ms` median, `latency_sigma`), `time_scale` (how much
of it is actually slept), TTFT as a uniform 15-35% of latency when streaming, token counts
as ceil(characters / 4), and judge-mode scoring by token F1 against the reference with
Gaussian noise plus `position_bias` for pairwise. In RAG mode the mock answers only if the
first 60 characters of an evidence passage appear in a numbered context block (see
RAG_EVALUATION.md).

`mock_seed` changes only the sampling noise: use it to simulate re-running the *same*
model. `difficulty_seed` changes which samples are hard: use it to simulate a *different*
model. Difficulty also depends on the model name, so two configurations with different
model names have independent difficulty draws, while a re-run with the same name and
`difficulty_seed` shares per-sample difficulty, which is what makes paired comparisons on
the mock behave like paired comparisons of real re-runs.

Mock numbers exercise the framework end to end with a known, controllable quality level.
They say nothing about any real model and are always labelled illustrative.

## 9. Limitations

- **Heuristic detectors.** Refusal, abstention, language and canary checks are substring
  or stop-word heuristics (sections 2.6-2.8). They are cheap and transparent but miss
  paraphrases and other languages; audit a sample of flagged and unflagged outputs.
- **Lexical embedder.** The default `HashEmbedder` makes `semantic_similarity`,
  `semantic_consistency` and RAG `answer_relevance` lexical-overlap measures. Configure a
  real embedding model before interpreting them as semantic.
- **Lexical answer matching.** Exact match, F1, ROUGE-L and `qa` pass reward surface
  overlap; correct paraphrases score low and wrong answers containing the reference phrase
  can pass.
- **Small datasets.** The bundled datasets have 25-40 items. With $n = 40$ the MDE is
  roughly 0.13-0.22 (section 4.5), and runs below 30 samples carry a warning. Adding
  trials does not substitute for adding samples (section 4.1).
- **Multiple runs over time.** Each comparison controls error for one pair of runs.
  Repeatedly comparing new candidates, gating many configurations or checking drift on
  many series increases the overall chance of a false alarm; corrections are applied only
  within one configuration pair.
- **Independence assumptions.** Tests treat samples as exchangeable units. Near-duplicate
  items, shared templates or cached outputs violate this and make intervals too narrow.
- **Judge biases.** Position, length and self-preference biases are mitigated
  (swapping, prompts, warnings), not eliminated. Judges must be calibrated against human
  labels in the target domain; the judge's own non-determinism adds variance that the
  bootstrap attributes to the candidate.
- **Pricing table.** Prices are user-maintained in `pricing.yaml` and never hardcoded.
  Cost numbers are only as accurate as that file and its `as_of` date; failed calls are
  excluded from `cost_usd` even if the provider billed them.
- **Mock provider.** It reads the reference through the oracle; its numbers validate the
  pipeline and statistics, not models.
- **Provider-side change.** A provider can change a model behind the same name; drift
  detection can notice a shift but cannot attribute it.
