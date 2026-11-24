// TypeScript mirrors of LLMBench's JSON outputs (pydantic models in src/llmbench).

export interface CI {
  estimate: number;
  low: number;
  high: number;
  level: number;
  method: string;
  n_samples: number;
  n_trials: number;
}

export interface MetricSummary {
  metric: string;
  higher_is_better: boolean;
  kind: string;
  ci: CI | null;
}

export interface Latency {
  n: number;
  mean: number | null;
  p50: number | null;
  p90: number | null;
  p95: number | null;
  p99: number | null;
  ttft_p50: number | null;
  ttft_p95: number | null;
  tokens_per_second: number | null;
  histogram: number[];
  bin_edges: number[];
}

export interface Cost {
  total_usd: number | null;
  per_1k_requests_usd: number | null;
  per_correct_usd: number | null;
  quality_per_dollar: number | null;
  cached_fraction: number;
}

export interface SampleRow {
  sample_id: string;
  pass_rate: number;
  input: string;
  reference: string;
  output: string;
  tags: string[];
}

export interface ConfigSummary {
  model_id: string;
  prompt_id: string;
  n_samples: number;
  n_trials: number;
  metrics: Record<string, MetricSummary>;
  latency: Latency;
  cost: Cost;
  failures: Record<string, number>;
  failure_rate: number;
  malformed_rate: number;
  refusal_rate: number;
  retry_distribution: Record<string, number>;
  tags: [string, number, number][];
  worst: SampleRow[];
}

export interface Calibration {
  judge: string;
  rubric: string;
  n: number;
  n_scored: number;
  accuracy: number | null;
  cohen_kappa: number | null;
  weighted_kappa_quadratic: number | null;
  spearman_scores: number | null;
  confusion: Record<string, Record<string, number>>;
  mean_self_consistency: number | null;
}

export interface RunSummary {
  run_id: string;
  name: string;
  status: string;
  task: string;
  created_at: string;
  dataset_name: string;
  dataset_hash: string;
  n_expected: number;
  n_done: number;
  total_cost_usd: number | null;
  stop_reason: string | null;
  manifest: {
    models?: { id: string; provider: string; model: string }[];
    config?: { trials?: number };
    [key: string]: unknown;
  };
  quality_metric: string;
  configs: ConfigSummary[];
  pareto: string[];
  pareto_objectives: string[];
  judge: {
    model?: string;
    calls?: number;
    malformed?: number;
    cost_usd?: number;
    candidate_overlap?: boolean;
    length_bias_spearman?: number | null;
  };
  calibration: Calibration | null;
  skipped_models: Record<string, string>;
}

export type Verdict = "improved" | "regressed" | "no significant change" | "insufficient data";

export interface MetricComparison {
  metric: string;
  higher_is_better: boolean;
  kind: string;
  n_samples: number;
  n_trials_baseline: number;
  n_trials_candidate: number;
  baseline: CI | null;
  candidate: CI | null;
  delta: number | null;
  delta_low: number | null;
  delta_high: number | null;
  relative_delta: number | null;
  test: string;
  p_value: number | null;
  p_adjusted: number | null;
  effect_size_name: string;
  effect_size: number | null;
  effect_magnitude: string | null;
  mde: number | null;
  alpha: number;
  verdict: Verdict;
  warning: string | null;
}

export interface SampleDiff {
  sample_id: string;
  baseline_score: number;
  candidate_score: number;
  baseline_output: string;
  candidate_output: string;
  tags: string[];
}

export interface TagBreakdown {
  tag: string;
  n_samples: number;
  baseline: number;
  candidate: number;
}

export interface GroupComparison {
  baseline_config: string;
  candidate_config: string;
  metrics: MetricComparison[];
  regressions: SampleDiff[];
  improvements: SampleDiff[];
  tags: TagBreakdown[];
  difficulty: TagBreakdown[];
}

export interface RunComparison {
  baseline_run: string;
  candidate_run: string;
  dataset_match: boolean;
  alpha: number;
  correction: string;
  groups: GroupComparison[];
  notes: string[];
}

export interface GateResult {
  metric: string;
  config: string;
  passed: boolean;
  reason: string;
  comparison: MetricComparison | null;
}

export interface GateReport {
  baseline_run: string;
  candidate_run: string;
  results: GateResult[];
  comparison: RunComparison;
  uses_mock: boolean;
}

export interface RagSetting {
  chunk_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  retrieval: string;
  reranker: string;
  embedder: string;
}

export interface SettingResult {
  setting: RagSetting;
  n_questions: number;
  metrics: Record<string, CI | null>;
  mean_latency_ms: number;
}

export interface SweepReport {
  name: string;
  generator: string;
  verifier: string;
  n_questions: number;
  results: SettingResult[];
  main_effects: Record<string, Record<string, Record<string, number>>>;
  best: Record<string, string>;
}

export interface DemoManifest {
  generated_at: string;
  commit: string | null;
  llmbench_version: string;
}

export interface RunInfo {
  run_id: string;
  name: string;
  status: string;
  task: string;
  dataset_hash: string;
  n_done: number;
  n_expected: number;
  total_cost_usd: number | null;
  created_at: string;
  finished_at: string | null;
  stop_reason: string | null;
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  run_id: string | null;
  run_status: string | null;
  done: number;
  total: number;
  error: string | null;
  submitted_at: string;
}

export const configKey = (c: { model_id: string; prompt_id: string }): string =>
  `${c.model_id}/${c.prompt_id}`;
