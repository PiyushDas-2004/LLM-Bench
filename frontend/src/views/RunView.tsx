import { useMemo, useState } from "react";
import type { ConfigSummary, RunSummary } from "../types";
import { configKey } from "../types";
import { label, ms, num, pct, smart, usd, when } from "../format";
import { CIBars, Histogram, ParetoScatter, type ScatterPoint } from "../components/charts";
import { Badge, Card, CICell, MockBanner, Segmented, Stat, Tabs } from "../components/ui";

const QUALITY_ORDER = [
  "pass",
  "llm_judge_correctness",
  "f1",
  "exact_match",
  "rouge_l",
  "semantic_similarity",
  "json_schema",
  "field_accuracy",
  "injection_resistance",
  "refusal_correctness",
  "abstention_correct",
];
const CONSISTENCY = ["exact_agreement", "semantic_consistency", "answer_entropy", "pass_at_k", "pass_hat_k", "flip_rate"];

/** Frontier of a 2-D projection: minimise x, maximise y. */
function frontier(points: { label: string; x: number; y: number }[]): ScatterPoint[] {
  return points.map((p) => ({
    ...p,
    front: !points.some((q) => q !== p && q.x <= p.x && q.y >= p.y && (q.x < p.x || q.y > p.y)),
  }));
}

export function RunView({ summary, isMock }: { summary: RunSummary; isMock: boolean }) {
  const configs = summary.configs;
  const present = useMemo(() => new Set(configs.flatMap((c) => Object.keys(c.metrics))), [configs]);
  const qualityMetrics = QUALITY_ORDER.filter((m) => present.has(m));
  const [metric, setMetric] = useState(qualityMetrics[0] ?? summary.quality_metric);
  const [xAxis, setXAxis] = useState<"cost" | "latency">(configs.every((c) => c.cost.per_1k_requests_usd !== null) ? "cost" : "latency");
  const [selected, setSelected] = useState(configKey(configs[0] ?? { model_id: "", prompt_id: "" }));
  const current: ConfigSummary | undefined = configs.find((c) => configKey(c) === selected) ?? configs[0];
  const trials = summary.manifest.config?.trials ?? Math.round((configs[0]?.n_trials ?? 0) / Math.max(configs[0]?.n_samples ?? 1, 1));

  const higherIsBetter = configs.find((x) => x.metrics[metric])?.metrics[metric]?.higher_is_better ?? true;
  const bars = configs
    .map((cfg) => ({ cfg, ci: cfg.metrics[metric]?.ci ?? null }))
    .filter((r) => r.ci)
    .sort((a, b) => (b.ci!.estimate - a.ci!.estimate) * (higherIsBetter ? 1 : -1))
    .map(({ cfg, ci }) => ({
      label: configKey(cfg),
      value: ci!.estimate,
      low: ci!.low,
      high: ci!.high,
      detail: `n = ${ci!.n_samples} samples × ${Math.round(ci!.n_trials / Math.max(ci!.n_samples, 1))} trials`,
    }));
  const qualityFor = (cfg: ConfigSummary) => cfg.metrics[summary.quality_metric]?.ci?.estimate;
  const points = frontier(
    configs
      .filter((cfg) => qualityFor(cfg) !== undefined && (xAxis === "cost" ? cfg.cost.per_1k_requests_usd !== null : cfg.latency.p50 !== null))
      .map((cfg) => ({
        label: configKey(cfg),
        x: xAxis === "cost" ? (cfg.cost.per_1k_requests_usd ?? 0) : (cfg.latency.p50 ?? 0),
        y: qualityFor(cfg) ?? 0,
      })),
  );
  const best = bars[0];

  return (
    <>
      {isMock && <MockBanner />}
      <div className="grid stats">
        <Stat k="Configurations" v={configs.length} s="model × prompt" />
        <Stat k="Trials" v={`${summary.n_done.toLocaleString()}`} s={`${trials} per sample · ${summary.status}`} />
        <Stat k={`Best ${label(metric)}`} v={best ? smart(best.value) : "n/a"} s={best?.label} />
        <Stat k={isMock ? "Simulated cost" : "Total cost"} v={usd(summary.total_cost_usd)} s={isMock ? "fictional mock prices" : undefined} />
        <Stat k="Dataset" v={summary.dataset_name} s={<span className="mono">{summary.dataset_hash.slice(0, 12)}</span>} />
      </div>

      <Card
        title="Leaderboard"
        sub="Point estimate with 95% cluster-bootstrap CI (samples resampled, then trials within samples). Click a bar to inspect that configuration."
        actions={
          <select className="btn" value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="metric">
            {qualityMetrics.map((m) => (
              <option key={m} value={m}>
                {label(m)}
              </option>
            ))}
          </select>
        }
      >
        <CIBars rows={bars} max={bars.every((b) => (b.high ?? b.value) <= 1) ? 1 : undefined} onSelect={setSelected} />
      </Card>

      <div className="grid cols-2">
        <Card
          title="Quality vs cost vs latency"
          sub={`Pareto frontier of ${label(summary.quality_metric)} against ${xAxis === "cost" ? "cost per 1k requests" : "p50 latency"}.`}
          actions={
            <Segmented
              items={[
                { id: "cost", label: "Cost" },
                { id: "latency", label: "Latency" },
              ]}
              value={xAxis}
              onChange={setXAxis}
            />
          }
        >
          <ParetoScatter
            points={points}
            xLabel={xAxis === "cost" ? "cost per 1k requests (USD)" : "p50 latency (ms)"}
            yLabel={label(summary.quality_metric)}
            xFmt={xAxis === "cost" ? (v) => `$${Number(v.toPrecision(2))}` : (v) => `${Math.round(v)}`}
          />
        </Card>
        <Card title="Latency distribution" sub="Successful calls only; dashed lines mark p50 and p95." actions={<ConfigPicker configs={configs} value={selected} onChange={setSelected} />}>
          {current && (
            <>
              <Histogram counts={current.latency.histogram} edges={current.latency.bin_edges} markers={{ p50: current.latency.p50, p95: current.latency.p95 }} />
              <dl className="kv" style={{ marginTop: 10 }}>
                <dt>p50 / p95 / p99</dt>
                <dd>
                  {ms(current.latency.p50)} / {ms(current.latency.p95)} / {ms(current.latency.p99)}
                </dd>
                <dt>Time to first token (p50)</dt>
                <dd>{ms(current.latency.ttft_p50)}</dd>
                <dt>Throughput</dt>
                <dd>{num(current.latency.tokens_per_second, 1)} tokens/s</dd>
              </dl>
            </>
          )}
        </Card>
      </div>

      <Card
        title="Consistency across trials"
        sub={`How stable is each configuration when the same input is sent ${trials} times? pass@k / pass^k use the unbiased estimators; entropy is in bits (0 = deterministic).`}
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                {CONSISTENCY.filter((m) => present.has(m)).map((m) => (
                  <th className="num" key={m}>
                    {label(m)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {configs.map((cfg) => (
                <tr key={configKey(cfg)}>
                  <td>{configKey(cfg)}</td>
                  {CONSISTENCY.filter((m) => present.has(m)).map((m) => (
                    <td className="num" key={m}>
                      <CICell ci={cfg.metrics[m]?.ci} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Reliability and cost">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th className="num">Failure rate</th>
                <th>Failures by class</th>
                <th className="num">Malformed</th>
                <th className="num">Refusals</th>
                <th>Attempts</th>
                <th className="num">Cost / 1k req</th>
                <th className="num">Cost / correct</th>
                <th className="num">Pareto</th>
              </tr>
            </thead>
            <tbody>
              {configs.map((cfg) => (
                <tr key={configKey(cfg)}>
                  <td>{configKey(cfg)}</td>
                  <td className="num">{pct(cfg.failure_rate)}</td>
                  <td>
                    {Object.entries(cfg.failures).length ? (
                      Object.entries(cfg.failures).map(([k, v]) => (
                        <span className="chip" key={k}>
                          {k} {v}
                        </span>
                      ))
                    ) : (
                      <span className="ci">none</span>
                    )}
                  </td>
                  <td className="num">{pct(cfg.malformed_rate)}</td>
                  <td className="num">{pct(cfg.refusal_rate)}</td>
                  <td>
                    {Object.entries(cfg.retry_distribution).map(([k, v]) => (
                      <span className="chip" key={k}>
                        {k}× {v}
                      </span>
                    ))}
                  </td>
                  <td className="num">{usd(cfg.cost.per_1k_requests_usd)}</td>
                  <td className="num">{usd(cfg.cost.per_correct_usd)}</td>
                  <td className="num">{summary.pareto.includes(configKey(cfg)) ? <span className="star">★</span> : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <TagsAndSamples configs={configs} selected={selected} onSelect={setSelected} metricName={summary.quality_metric} />

      {summary.judge.model && (
        <Card title="LLM judge" sub="Every judge call is stored for audit.">
          <dl className="kv">
            <dt>Judge</dt>
            <dd>
              <span className="mono">{summary.judge.model}</span>{" "}
              {summary.judge.candidate_overlap && <Badge tone="bad">also a candidate: self-preference risk</Badge>}
            </dd>
            <dt>Calls</dt>
            <dd>
              {summary.judge.calls} ({summary.judge.malformed} malformed)
            </dd>
            <dt>Length bias</dt>
            <dd>Spearman ρ(answer length, score) = {num(summary.judge.length_bias_spearman ?? null)}</dd>
            {summary.calibration && (
              <>
                <dt>Calibration</dt>
                <dd>
                  Cohen's κ {num(summary.calibration.cohen_kappa)} · accuracy {pct(summary.calibration.accuracy)} on {summary.calibration.n_scored} human-labelled
                  examples
                </dd>
              </>
            )}
          </dl>
        </Card>
      )}

      <details className="card">
        <summary>Run manifest (everything needed to reproduce this run)</summary>
        <p className="card-sub">
          Run <span className="mono">{summary.run_id}</span> · created {when(summary.created_at)}
        </p>
        <pre className="code">{JSON.stringify(summary.manifest, null, 2)}</pre>
      </details>
    </>
  );
}

function ConfigPicker({ configs, value, onChange }: { configs: ConfigSummary[]; value: string; onChange: (v: string) => void }) {
  return (
    <select className="btn" value={value} onChange={(e) => onChange(e.target.value)} aria-label="configuration">
      {configs.map((c) => (
        <option key={configKey(c)} value={configKey(c)}>
          {configKey(c)}
        </option>
      ))}
    </select>
  );
}

function TagsAndSamples({ configs, selected, onSelect, metricName }: { configs: ConfigSummary[]; selected: string; onSelect: (v: string) => void; metricName: string }) {
  const [tab, setTab] = useState<"tags" | "worst">("tags");
  const tags = [...new Set(configs.flatMap((c) => c.tags.map((t) => t[0])))].sort();
  const current = configs.find((c) => configKey(c) === selected) ?? configs[0];
  return (
    <Card
      title={tab === "tags" ? `Per-tag ${label(metricName)}` : "Worst-performing samples"}
      actions={
        <Tabs
          items={[
            { id: "tags", label: "By tag" },
            { id: "worst", label: "Worst samples" },
          ]}
          value={tab}
          onChange={setTab}
        />
      }
    >
      {tab === "tags" ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Tag</th>
                {configs.map((c) => (
                  <th className="num" key={configKey(c)}>
                    {configKey(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tags.map((tag) => (
                <tr key={tag}>
                  <td>{tag}</td>
                  {configs.map((c) => {
                    const row = c.tags.find((t) => t[0] === tag);
                    return (
                      <td className="num" key={configKey(c)}>
                        {row ? (
                          <>
                            {num(row[2])} <span className="ci">n={row[1]}</span>
                          </>
                        ) : (
                          "n/a"
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div style={{ marginBottom: 10 }}>
            <ConfigPicker configs={configs} value={configKey(current ?? configs[0]!)} onChange={onSelect} />
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Sample</th>
                  <th className="num">Pass rate</th>
                  <th>Input</th>
                  <th>Reference</th>
                  <th>Output (trial 0)</th>
                </tr>
              </thead>
              <tbody>
                {(current?.worst ?? []).map((w) => (
                  <tr key={w.sample_id}>
                    <td>
                      {w.sample_id}
                      <br />
                      {w.tags.map((t) => (
                        <span className="chip" key={t}>
                          {t}
                        </span>
                      ))}
                    </td>
                    <td className="num">{pct(w.pass_rate, 0)}</td>
                    <td className="out">{w.input}</td>
                    <td className="out">{w.reference}</td>
                    <td className="out">{w.output}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}
