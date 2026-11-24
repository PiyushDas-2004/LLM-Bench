import type { GateReport } from "../types";
import { Badge, Card, MockBanner } from "../components/ui";
import { ComparisonPanel } from "./CompareView";

export function GateView({ report }: { report: GateReport }) {
  const passed = report.results.every((r) => r.passed);
  return (
    <>
      {report.uses_mock && <MockBanner />}
      <div className={`card`} style={{ borderLeft: `5px solid var(${passed ? "--good" : "--bad"})` }}>
        <div className="card-head">
          <h2 style={{ fontSize: 20 }}>Eval gate {passed ? <Badge tone="good">✓ PASSED</Badge> : <Badge tone="bad">✗ FAILED</Badge>}</h2>
          <span className="card-sub">exit code {passed ? 0 : 1}</span>
        </div>
        <p className="card-sub" style={{ margin: 0 }}>
          A gate fails only when the metric moves in the bad direction by more than its tolerance <b>and</b> the change is
          statistically significant after multiple-comparison correction. This demo re-runs the configuration with a seeded
          regression (mock accuracy 0.85 → 0.40) against the stored baseline in <span className="mono">baselines/gate-demo.json</span>.
        </p>
      </div>
      <Card title="Gate results">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Gate</th>
                <th>Configuration</th>
                <th>Result</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {report.results.map((r) => (
                <tr key={`${r.metric}-${r.config}`}>
                  <td>{r.metric}</td>
                  <td className="mono">{r.config}</td>
                  <td>{r.passed ? <Badge tone="good">✓ pass</Badge> : <Badge tone="bad">✗ fail</Badge>}</td>
                  <td>{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <ComparisonPanel comparison={report.comparison} />
    </>
  );
}
