export const num = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toFixed(digits);

/** 3 decimals, or 3 significant digits for tiny magnitudes such as per-call cost. */
export const smart = (v: number | null | undefined, sign = false): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  if (Math.abs(v) < 1e-12) return (0).toFixed(3);
  const s = sign && v > 0 ? "+" : "";
  if (v !== 0 && Math.abs(v) < 0.01) return s + v.toPrecision(3);
  if (Math.abs(v) >= 1000) return s + v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return s + v.toFixed(3);
};

export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined ? "n/a" : `${(v * 100).toFixed(digits)}%`;

export const ms = (v: number | null | undefined): string =>
  v === null || v === undefined ? "n/a" : `${Math.round(v).toLocaleString()} ms`;

export const usd = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return "unknown";
  if (v === 0) return "$0";
  if (v < 0.01) return `$${v.toPrecision(2)}`;
  return v < 1 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
};

export const pval = (v: number | null | undefined): string =>
  v === null || v === undefined ? "n/a" : v < 0.001 ? "<0.001" : v.toFixed(3);

export const when = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? iso
    : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};

export const label = (s: string): string => s.replace(/_/g, " ");

/** "Nice" tick values (1, 2, 2.5, 5 x 10^k steps) covering [lo, hi]. */
export function ticks(lo: number, hi: number, n = 4): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    out.push(Number(v.toFixed(10)));
  }
  return out;
}

export const tickLabel = (v: number): string => {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 1) return String(Number(v.toFixed(2)));
  return String(Number(v.toPrecision(2)));
};
