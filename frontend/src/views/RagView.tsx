import { useMemo, useState } from "react";
import type { SettingResult, SweepReport } from "../types";
import { label, num } from "../format";
import { LevelBars } from "../components/charts";
import { Card, CICell, MockBanner, Stat } from "../components/ui";

const METRICS = [
  "pass",
  "answer_f1",
  "context_recall",
  "context_precision",
  "mrr",
  "ndcg_at_k",
  "faithfulness",
  "hallucination",
  "citation_accuracy",
  "abstention_correct",
  "answer_relevance",
];
const LOWER_IS_BETTER = new Set(["hallucination"]);

const settingLabel = (r: SettingResult): string => {
  const s = r.setting;
  return `${s.chunk_strategy}/${s.chunk_size}/${s.chunk_overlap} · ${s.retrieval} · k=${s.top_k} · rerank=${s.reranker}`;
};

export function RagView({ report }: { report: SweepReport }) {
  const [metric, setMetric] = useState("pass");
  const lower = LOWER_IS_BETTER.has(metric);
  const ranked = useMemo(
    () =>
      [...report.results]
        .filter((r) => r.metrics[metric])
        .sort((a, b) => ((b.metrics[metric]?.estimate ?? 0) - (a.metrics[metric]?.estimate ?? 0)) * (lower ? -1 : 1)),
    [report.results, metric, lower],
  );
  const factors = Object.entries(report.main_effects).filter(([, perMetric]) => perMetric[metric]);
  const spreads = factors
    .map(([factor, perMetric]) => {
      const vals = Object.values(perMetric[metric] ?? {});
      return { factor, spread: Math.max(...vals) - Math.min(...vals) };
    })
    .sort((a, b) => b.spread - a.spread);
  return (
    <>
      <MockBanner />
      <div className="grid stats">
        <Stat k="Settings evaluated" v={report.results.length} s="Cartesian product of the sweep grid" />
        <Stat k="Questions" v={report.n_questions} s="rag_mini, incl. unanswerable ones" />
        <Stat k="Generator" v={<span style={{ fontSize: 16 }}>{report.generator}</span>} s={`verifier: ${report.verifier}`} />
        <Stat k={`Most influential factor for ${label(metric)}`} v={<span style={{ fontSize: 18 }}>{spreads[0] ? label(spreads[0].factor) : "n/a"}</span>} s={spreads[0] ? `spread ${num(spreads[0].spread)}` : undefined} />
      </div>
      <Card
        title="Which setting moves which metric"
        sub="Main effects: the mean of the metric across all settings at each level of a factor. Factors are ordered by how much they move the metric."
        actions={
          <select className="btn" value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="metric">
            {METRICS.map((m) => (
              <option key={m} value={m}>
                {label(m)}
              </option>
            ))}
          </select>
        }
      >
        <div className="grid cols-2">
          {spreads.map(({ factor }) => (
            <div key={factor}>
              <h3 style={{ fontSize: 14, margin: "4px 0" }}>{label(factor)}</h3>
              <LevelBars levels={report.main_effects[factor]?.[metric] ?? {}} metric={label(metric)} lowerIsBetter={lower} />
            </div>
          ))}
        </div>
      </Card>
      <Card title={`Top settings by ${label(metric)}`} sub="Each value: estimate [95% bootstrap CI over questions].">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Setting</th>
                {["pass", "answer_f1", "context_recall", "mrr", "faithfulness", "abstention_correct"].map((m) => (
                  <th className="num" key={m}>
                    {label(m)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ranked.slice(0, 15).map((r, i) => (
                <tr key={settingLabel(r)}>
                  <td>{i + 1}</td>
                  <td className="mono">{settingLabel(r)}</td>
                  {["pass", "answer_f1", "context_recall", "mrr", "faithfulness", "abstention_correct"].map((m) => (
                    <td className="num" key={m}>
                      <CICell ci={r.metrics[m]} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
