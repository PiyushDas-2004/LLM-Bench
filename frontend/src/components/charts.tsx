import { useEffect, useRef, useState, type RefObject } from "react";
import type { MetricComparison } from "../types";
import { pval, smart, tickLabel, ticks } from "../format";
import { useTooltip } from "./ui";

// All charts are plain SVG drawn at the container's real pixel width (so text stays a
// constant size), with CSS-token colours and hover tooltips.

function useWidth(fallback: number): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setWidth(Math.max(260, Math.floor(el.getBoundingClientRect().width)));
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export interface BarRow {
  label: string;
  value: number;
  low?: number | null;
  high?: number | null;
  highlight?: boolean;
  detail?: string;
}

export function CIBars({ rows, max, unit = "", onSelect }: { rows: BarRow[]; max?: number; unit?: string; onSelect?: (label: string) => void }) {
  const tip = useTooltip();
  const [ref, width] = useWidth(640);
  if (!rows.length) return null;
  const labelW = Math.min(170, Math.max(90, width * 0.28));
  const plotW = width - labelW - 70;
  const rowH = 34;
  const top = 8;
  const height = top + rows.length * rowH + 28;
  const vmax = max ?? Math.max(...rows.map((r) => r.high ?? r.value), 1e-9) * 1.05;
  const sx = (v: number) => labelW + (plotW * Math.max(0, Math.min(v, vmax))) / vmax;
  return (
    <div ref={ref}>
    <svg className="chart" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="bar chart with confidence intervals">
      {ticks(0, vmax).map((t) => (
        <g key={t}>
          <line className="grid-line" x1={sx(t)} x2={sx(t)} y1={top} y2={height - 22} />
          <text x={sx(t)} y={height - 6} textAnchor="middle">
            {tickLabel(t)}
            {unit}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y = top + i * rowH;
        const hasCI = r.low !== null && r.low !== undefined && r.high !== null && r.high !== undefined;
        return (
          <g
            key={r.label}
            className="mark"
            style={{ cursor: onSelect ? "pointer" : "default" }}
            onClick={() => onSelect?.(r.label)}
            onMouseMove={(e) =>
              tip.show(
                e,
                <>
                  <b>{r.label}</b>
                  <br />
                  {smart(r.value)}
                  {unit}
                  {hasCI && (
                    <>
                      {" "}
                      (95% CI {smart(r.low)}–{smart(r.high)})
                    </>
                  )}
                  {r.detail && (
                    <>
                      <br />
                      {r.detail}
                    </>
                  )}
                </>,
              )
            }
            onMouseLeave={tip.hide}
          >
            <rect className="hit" x={0} y={y} width={width} height={rowH} />
            <text className="label" x={labelW - 10} y={y + 20} textAnchor="end">
              {r.label}
            </text>
            <rect className={`bar ${r.highlight === false ? "dim" : ""}`} x={labelW} y={y + 8} width={Math.max(sx(r.value) - labelW, 2)} height={16} rx={4} />
            {hasCI && (
              <g>
                <line className="whisker" x1={sx(r.low!)} x2={sx(r.high!)} y1={y + 16} y2={y + 16} />
                <line className="whisker" x1={sx(r.low!)} x2={sx(r.low!)} y1={y + 11} y2={y + 21} />
                <line className="whisker" x1={sx(r.high!)} x2={sx(r.high!)} y1={y + 11} y2={y + 21} />
              </g>
            )}
            <text className="value" x={labelW + plotW + 8} y={y + 20}>
              {smart(r.value)}
            </text>
          </g>
        );
      })}
    </svg>
    </div>
  );
}

export interface ScatterPoint {
  label: string;
  x: number;
  y: number;
  front: boolean;
}

export function ParetoScatter({ points, xLabel, yLabel, xFmt = tickLabel }: { points: ScatterPoint[]; xLabel: string; yLabel: string; xFmt?: (v: number) => string }) {
  const tip = useTooltip();
  const [ref, width] = useWidth(640);
  if (!points.length) return null;
  const W = width;
  const H = Math.round(Math.min(380, Math.max(260, width * 0.6)));
  const L = 56;
  const R = 24;
  const T = 18;
  const B = 46;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.12 || Math.abs(xs[0] ?? 1) * 0.1 || 1;
  const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.18 || 0.05;
  const x0 = Math.max(0, Math.min(...xs) - xPad);
  const x1 = Math.max(...xs) + xPad;
  const y0 = Math.max(0, Math.min(...ys) - yPad);
  const y1 = Math.min(1, Math.max(...ys) + yPad) > y0 ? Math.min(1, Math.max(...ys) + yPad) : y0 + 0.1;
  const sx = (v: number) => L + ((W - L - R) * (v - x0)) / (x1 - x0);
  const sy = (v: number) => H - B - ((H - T - B) * (v - y0)) / (y1 - y0);
  const front = points.filter((p) => p.front).sort((a, b) => a.x - b.x);
  return (
    <div ref={ref}>
      <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} versus ${xLabel}`}>
        {ticks(y0, y1).map((t) => (
          <g key={`y${t}`}>
            <line className="grid-line" x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} />
            <text x={L - 8} y={sy(t) + 4} textAnchor="end">
              {tickLabel(t)}
            </text>
          </g>
        ))}
        {ticks(x0, x1).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - B + 18} textAnchor="middle">
            {xFmt(t)}
          </text>
        ))}
        <line className="axis-line" x1={L} x2={W - R} y1={H - B} y2={H - B} />
        <text className="label" x={(L + W - R) / 2} y={H - 6} textAnchor="middle">
          {xLabel}
        </text>
        <text className="label" transform={`translate(14 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">
          {yLabel}
        </text>
        {front.length > 1 && <path className="frontier" d={front.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ")} />}
        {[...points].sort((a, b) => Number(a.front) - Number(b.front)).map((p) => {
          const right = sx(p.x) > W * 0.68;
          return (
            <g
              key={p.label}
              className="mark"
              onMouseMove={(e) =>
                tip.show(
                  e,
                  <>
                    <b>{p.label}</b>
                    <br />
                    {yLabel}: {smart(p.y)}
                    <br />
                    {xLabel}: {xFmt(p.x)}
                    <br />
                    {p.front ? "On the Pareto frontier" : "Dominated in this view"}
                  </>,
                )
              }
              onMouseLeave={tip.hide}
            >
              <circle className={`dot ${p.front ? "" : "dominated"}`} cx={sx(p.x)} cy={sy(p.y)} r={7} />
              <circle className="hit" cx={sx(p.x)} cy={sy(p.y)} r={16} />
              <text className="label" x={sx(p.x) + (right ? -11 : 11)} y={sy(p.y) - 10} textAnchor={right ? "end" : "start"}>
                {p.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span>
          <i className="swatch" style={{ background: "var(--series-1)" }} /> Pareto-optimal in this view
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--dominated)" }} /> dominated
        </span>
      </div>
    </div>
  );
}

export function Histogram({ counts, edges, markers }: { counts: number[]; edges: number[]; markers: Record<string, number | null> }) {
  const tip = useTooltip();
  const [ref, width] = useWidth(360);
  if (!counts.length || edges.length < 2) return <div className="empty">No latency data</div>;
  const W = width;
  const H = 170;
  const L = 6;
  const B = 24;
  const T = 20;
  const lo = edges[0] ?? 0;
  const hi = edges[edges.length - 1] ?? 1;
  const span = hi - lo || 1;
  const peak = Math.max(...counts, 1);
  const bw = (W - 2 * L) / counts.length;
  const sx = (v: number) => L + ((W - 2 * L) * (v - lo)) / span;
  return (
    <div ref={ref}>
    <svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="latency histogram">
      {counts.map((c, i) => {
        const h = ((H - T - B) * c) / peak;
        return (
          <g
            key={i}
            className="mark"
            onMouseMove={(e) => tip.show(e, <>{`${Math.round(edges[i] ?? 0)}–${Math.round(edges[i + 1] ?? 0)} ms: `}<b>{c}</b> calls</>)}
            onMouseLeave={tip.hide}
          >
            <rect className="hit" x={L + i * bw} y={T} width={bw} height={H - T - B} />
            <rect className="bar" x={L + i * bw + 1} y={H - B - h} width={Math.max(bw - 2, 1)} height={Math.max(h, 0.5)} rx={2} />
          </g>
        );
      })}
      <line className="axis-line" x1={L} x2={W - L} y1={H - B} y2={H - B} />
      {Object.entries(markers).map(([name, v]) =>
        v === null ? null : (
          <g key={name}>
            <line className="zero-line" x1={sx(v)} x2={sx(v)} y1={T - 4} y2={H - B} />
            <text x={sx(v)} y={T - 7} textAnchor="middle">
              {name}
            </text>
          </g>
        ),
      )}
      <text x={L} y={H - 6}>
        {Math.round(lo)} ms
      </text>
      <text x={W - L} y={H - 6} textAnchor="end">
        {Math.round(hi)} ms
      </text>
    </svg>
    </div>
  );
}

/** Forest plot: each metric's paired difference with its 95% CI, on its own symmetric scale. */
export function ForestPlot({ rows }: { rows: MetricComparison[] }) {
  const tip = useTooltip();
  const [ref, width] = useWidth(720);
  const usable = rows.filter((r) => r.delta !== null && r.delta_low !== null && r.delta_high !== null);
  if (!usable.length) return <div className="empty">No comparable metrics</div>;
  const labelW = Math.min(190, Math.max(110, width * 0.25));
  const plotW = Math.min(520, width - labelW - 120);
  const rowH = 36;
  const H = usable.length * rowH + 30;
  const cx = labelW + plotW / 2;
  return (
    <div ref={ref}>
      <svg className="chart" width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="img" aria-label="forest plot of metric differences">
        <line className="zero-line" x1={cx} x2={cx} y1={4} y2={H - 22} />
        <text x={cx} y={H - 6} textAnchor="middle">
          0 (no change)
        </text>
        <text x={labelW + 6} y={H - 6}>
          ← lower
        </text>
        <text x={labelW + plotW - 6} y={H - 6} textAnchor="end">
          higher →
        </text>
        {usable.map((r, i) => {
          const extent = Math.max(Math.abs(r.delta_low!), Math.abs(r.delta_high!), Math.abs(r.delta!), 1e-12) * 1.15;
          const sx = (v: number) => cx + (plotW / 2) * (v / extent);
          const y = 6 + i * rowH;
          const tone = r.verdict === "improved" ? "good" : r.verdict === "regressed" ? "bad" : "neutral";
          return (
            <g
              key={r.metric}
              className="mark"
              onMouseMove={(e) =>
                tip.show(
                  e,
                  <>
                    <b>{r.metric}</b> ({r.higher_is_better ? "higher" : "lower"} is better)
                    <br />Δ {smart(r.delta, true)} [{smart(r.delta_low, true)}, {smart(r.delta_high, true)}]
                    <br />
                    {r.test}, p(adj) {pval(r.p_adjusted)}
                    <br />
                    {r.effect_size_name} {smart(r.effect_size)} · MDE {smart(r.mde)}
                  </>,
                )
              }
              onMouseLeave={tip.hide}
            >
              <rect className="hit" x={0} y={y} width={width} height={rowH} />
              <text className="label" x={labelW - 10} y={y + 20} textAnchor="end">
                {r.metric}
              </text>
              <line className={`${tone}-stroke`} strokeWidth={2.5} x1={sx(r.delta_low!)} x2={sx(r.delta_high!)} y1={y + 16} y2={y + 16} />
              <circle className={`${tone}-fill`} cx={sx(r.delta!)} cy={y + 16} r={5.5} />
              <text className="value" x={labelW + plotW + 10} y={y + 20}>
                {smart(r.delta, true)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span><i className="swatch" style={{ background: "var(--good)" }} /> improved</span>
        <span><i className="swatch" style={{ background: "var(--bad)" }} /> regressed</span>
        <span><i className="swatch" style={{ background: "var(--muted)" }} /> no significant change</span>
        <span>Each row uses its own scale, centred on zero; whether higher is better depends on the metric.</span>
      </div>
    </div>
  );
}

/** Mean of a metric at each level of one factor (RAG sweep main effects). */
export function LevelBars({ levels, metric, lowerIsBetter = false }: { levels: Record<string, number>; metric: string; lowerIsBetter?: boolean }) {
  const rows = Object.entries(levels).map(([label, value]) => ({ label, value }));
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  const better = (a: number, b: number) => (lowerIsBetter ? a < b : a > b);
  const best = rows.reduce((a, b) => (better(b.value, a.value) ? b : a), rows[0] ?? { label: "", value: 0 });
  return (
    <CIBars
      rows={rows.map((r) => ({ ...r, highlight: r.label === best.label, detail: `mean ${metric} across settings at this level` }))}
      max={max <= 1 ? 1 : max * 1.05}
    />
  );
}
