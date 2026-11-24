import type { Calibration } from "../types";
import { num, pct } from "../format";
import { Card, MockBanner, Stat } from "../components/ui";

export function CalibrationView({ report }: { report: Calibration }) {
  const total = Object.values(report.confusion).reduce((s, row) => s + Object.values(row).reduce((a, b) => a + b, 0), 0) || 1;
  const cell = (human: string, judge: string) => {
    const value = report.confusion[human]?.[judge] ?? 0;
    const agree = human === judge;
    const alpha = 0.12 + 0.75 * (value / total);
    return (
      <div
        className="cell"
        style={{ background: `color-mix(in srgb, var(${agree ? "--accent" : "--bad"}) ${Math.round(alpha * 100)}%, var(--surface))` }}
        title={`human ${human}, judge ${judge}: ${value}`}
      >
        {value}
      </div>
    );
  };
  return (
    <>
      <MockBanner />
      <div className="grid stats">
        <Stat k="Cohen's κ (pass / fail)" v={num(report.cohen_kappa)} s="0 = chance agreement, 1 = perfect" />
        <Stat k="Weighted κ (1–5 scores)" v={num(report.weighted_kappa_quadratic)} s="quadratic weights" />
        <Stat k="Accuracy" v={pct(report.accuracy)} s={`${report.n_scored} of ${report.n} labelled examples`} />
        <Stat k="Spearman (scores)" v={num(report.spearman_scores)} s="judge score vs human score" />
      </div>
      <div className="grid cols-2">
        <Card title="Confusion matrix" sub="Rows: human label · columns: judge label">
          <div className="matrix">
            <div />
            <div className="hd">judge: pass</div>
            <div className="hd">judge: fail</div>
            <div className="hd">human: pass</div>
            {cell("pass", "pass")}
            {cell("pass", "fail")}
            <div className="hd">human: fail</div>
            {cell("fail", "pass")}
            {cell("fail", "fail")}
          </div>
        </Card>
        <Card title="How to read this">
          <p style={{ marginTop: 0 }}>
            Before trusting an LLM judge, LLMBench scores it against human labels. The judge here is the mock judge
            (<span className="mono">{report.judge}</span>, rubric <span className="mono">{report.rubric}</span>), which grades by word overlap with the
            reference. It rejects many correct but verbose answers, which is the off-diagonal "human pass, judge fail" cell.
          </p>
          <p style={{ marginBottom: 0 }}>
            Moderate κ like this is the signal to improve the rubric or switch judge model before using the judge as a quality
            gate. Calibration exists to catch exactly this.
          </p>
        </Card>
      </div>
    </>
  );
}
