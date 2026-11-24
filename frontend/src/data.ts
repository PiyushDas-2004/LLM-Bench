import type {
  DemoManifest,
  GateReport,
  JobStatus,
  RunComparison,
  RunInfo,
  RunSummary,
  SweepReport,
  Calibration,
} from "./types";

// ---------------------------------------------------------------- demo data
// Static JSON produced by scripts/build_site.sh in CI (mock provider, no cost).

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { error?: string; detail?: unknown };
      if (body.error) detail = body.error;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      // body was not JSON; keep the status text
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const demo = {
  manifest: () => getJson<DemoManifest>("./data/manifest.json"),
  run: (name: "quickstart" | "safety") => getJson<RunSummary>(`./data/${name}.json`),
  comparison: () => getJson<RunComparison>("./data/comparison.json"),
  gate: () => getJson<GateReport>("./data/gate.json"),
  sweep: () => getJson<SweepReport>("./data/rag_sweep.json"),
  calibration: () => getJson<Calibration>("./data/calibration.json"),
};

// ---------------------------------------------------------------- live API

export interface Connection {
  baseUrl: string;
  apiKey: string;
}

const STORAGE_KEY = "llmbench.connection";

export function loadConnection(): Connection | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Connection) : null;
  } catch {
    return null;
  }
}

export function saveConnection(conn: Connection | null): void {
  try {
    if (conn) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(conn));
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // storage unavailable (private mode); the connection lasts for this page only
  }
}

export class LiveApi {
  constructor(private readonly conn: Connection) {}

  private url(path: string): string {
    return `${this.conn.baseUrl.replace(/\/+$/, "")}${path}`;
  }

  private headers(json = false): HeadersInit {
    return { "X-API-Key": this.conn.apiKey, ...(json ? { "Content-Type": "application/json" } : {}) };
  }

  health(): Promise<{ status: string; version: string }> {
    return getJson(this.url("/health"));
  }

  runs(): Promise<RunInfo[]> {
    return getJson(this.url("/runs?limit=100"), { headers: this.headers() });
  }

  results(ref: string): Promise<RunSummary> {
    return getJson(this.url(`/runs/${encodeURIComponent(ref)}/results`), { headers: this.headers() });
  }

  job(id: string): Promise<JobStatus> {
    return getJson(this.url(`/runs/${encodeURIComponent(id)}`), { headers: this.headers() });
  }

  submit(config: object): Promise<JobStatus> {
    return getJson(this.url("/runs"), {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify({ config, use_cache: true }),
    });
  }

  compare(baseline: string, candidate: string): Promise<RunComparison> {
    return getJson(this.url("/compare"), {
      method: "POST",
      headers: this.headers(true),
      body: JSON.stringify({ baseline, candidate }),
    });
  }
}
