import { useState } from "react";
import type { GroupComparison, RunComparison } from "../types";
import { num, pval, smart } from "../format";
import { ForestPlot } from "../components/charts";
import { Card, VerdictBadge } from "../components/ui";

export function ComparisonPanel({ comparison }: { comparison: RunComparison }) {
  const [index, setIndex] = useState(0);
  const group: GroupComparison | undefined = comparison.groups[index] ?? comparison.groups[0];
  if (!group) return <div className="empty">No comparable configurations.</div>;
  const regressed = group.metrics.filter((m) => m.verdict === "regressed").length;
  const improved = group.metrics.filter((m) => m.verdict === "improved").length;
  return (
    <>
      {comparison.groups.length > 1 && (
        <div className="seg" role="radiogroup" aria-label="configuration pair">
          {comparison.groups.map((g, i) => (
            <button key={`${g.baseline_config}->${g.candidate_config}`} className={i === index ? "on" : ""} onClick={() => setIndex(i)}>
              {g.baseline_config} → {g.candidate_config}
            </button>
          ))}
        </div>
      )}
      {comparison.notes.map((n) => (
        <div className="banner" key={n}>
          {n}
        </div>
      ))}
      <div className="grid stats">
        <div className="card stat">
          <div className="k">Baseline</div>
          <div className="v" style={{ fontSize: 18 }}>{group.baseline_config}</div>
        </div>
        <div className="card stat">
          <div className="k">Candidate</div>
          <div className="v" style={{ fontSize: 18 }}>{group.candidate_config}</div>
        </div>
        <div className="card stat">
          <div className="k">Significant changes</div>
          <div className="v">
            {improved} ▲ / {regressed} ▼
          </div>
          <div className="s">
            of {group.metrics.length} metrics · α = {comparison.alpha}, {comparison.correction} correction
          </div>
        </div>
      </div>
      <Card title="Paired differences (candidate − baseline)" sub="Dot = mean paired difference over aligned samples, bar = 95% bootstrap CI. Hover for the test, adjusted p-value, effect size and minimum detectable effect.">
        <ForestPlot rows={group.metrics} />
      </Card>
      <Card title="Statistical detail">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Baseline</th>
                <th className="num">Candidate</th>
                <th className="num">Δ [95% CI]</th>
                <th>Test</th>
                <th className="num">p (adj.)</th>
                <th className="num">Effect size</th>
                <th className="num">MDE</th>
                <th className="num">n</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {group.metrics.map((m) => (
                <tr key={m.metric}>
                  <td>{m.metric}</td>
                  <td className="num">{smart(m.baseline?.estimate)}</td>
                  <td className="num">{smart(m.candidate?.estimate)}</td>
                  <td className="num">
                    {smart(m.delta, true)} <span className="ci">[{smart(m.delta_low, true)}, {smart(m.delta_high, true)}]</span>
                  </td>
                  <td>{m.test.replace("paired-", "")}</td>
                  <td className="num">{pval(m.p_adjusted)}</td>
                  <td className="num">
                    {smart(m.effect_size)} <span className="ci">{m.effect_size_name}</span>
                  </td>
                  <td className="num">{smart(m.mde)}</td>
                  <td className="num">
                    {m.n_samples}×{Math.round(m.n_trials_candidate / Math.max(m.n_samples, 1))}
                  </td>
                  <td>
                    <VerdictBadge verdict={m.verdict} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid cols-2">
        <Card title="Per-sample regressions" sub="Pass rate across trials, with the first trial's output side by side.">
          {group.regressions.length ? (
            <SampleTable rows={group.regressions} />
          ) : (
            <div className="empty">No sample got worse.</div>
          )}
        </Card>
        <Card title="By tag">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tag</th>
                  <th className="num">n</th>
                  <th className="num">Baseline</th>
                  <th className="num">Candidate</th>
                  <th className="num">Δ</th>
                </tr>
              </thead>
              <tbody>
                {group.tags.map((t) => (
                  <tr key={t.tag}>
                    <td>{t.tag}</td>
                    <td className="num">{t.n_samples}</td>
                    <td className="num">{num(t.baseline)}</td>
                    <td className="num">{num(t.candidate)}</td>
                    <td className="num">{smart(t.candidate - t.baseline, true)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </>
  );
}

function SampleTable({ rows }: { rows: GroupComparison["regressions"] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Sample</th>
            <th className="num">Before → after</th>
            <th>Baseline output</th>
            <th>Candidate output</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.sample_id}>
              <td>{d.sample_id}</td>
              <td className="num">
                {num(d.baseline_score, 2)} → {num(d.candidate_score, 2)}
              </td>
              <td className="out">{d.baseline_output}</td>
              <td className="out">{d.candidate_output}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
