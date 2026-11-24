import { useCallback, useEffect, useRef, useState } from "react";
import type { JobStatus, RunComparison, RunInfo, RunSummary } from "../types";
import { LiveApi, loadConnection, saveConnection, type Connection } from "../data";
import { usd, when } from "../format";
import { Badge, Card, PageHead } from "../components/ui";
import { RunView } from "./RunView";
import { ComparisonPanel } from "./CompareView";

type Preset = "qa" | "classification" | "safety";

const PRESETS: Record<Preset, { label: string; dataset: string; task: string; prompt: { template: string; system_template?: string }; metrics: string[] }> = {
  qa: { label: "Factual QA", dataset: "datasets/qa_basic.jsonl", task: "qa", prompt: { template: "prompts/qa_v1.j2" }, metrics: ["exact_match", "f1", "consistency", "latency", "cost"] },
  classification: {
    label: "Intent classification",
    dataset: "datasets/classification_intent.jsonl",
    task: "classification",
    prompt: { template: "prompts/classification_v1.j2" },
    metrics: ["exact_match", "consistency", "latency"],
  },
  safety: {
    label: "Prompt-injection safety",
    dataset: "datasets/safety_injection.jsonl",
    task: "safety",
    prompt: { template: "prompts/safety_v1.j2", system_template: "prompts/safety_system.j2" },
    metrics: ["safety", "latency"],
  },
};

function buildConfig(preset: Preset, samples: number, trials: number, accA: number, accB: number, temperature: number) {
  const p = PRESETS[preset];
  const model = (id: string, accuracy: number, latency: number) => ({
    id,
    provider: "mock",
    model: id,
    params: { accuracy, temperature, latency_ms: latency, stream: true, injection_susceptibility: 1 - accuracy },
  });
  return {
    name: `web-${preset}`,
    dataset: p.dataset,
    task: p.task,
    trials,
    seed: 7,
    sampling: { n: samples, seed: 0 },
    pricing: "examples/pricing.yaml",
    models: [model("mock-strong", accA, 600), model("mock-fast", accB, 180)],
    prompts: [{ id: "v1", ...p.prompt }],
    metrics: p.metrics,
  };
}

export function LiveView() {
  const [conn, setConn] = useState<Connection | null>(loadConnection);
  const [draft, setDraft] = useState<Connection>(conn ?? { baseUrl: "", apiKey: "" });
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [comparison, setComparison] = useState<RunComparison | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [preset, setPreset] = useState<Preset>("qa");
  const [samples, setSamples] = useState(20);
  const [trials, setTrials] = useState(3);
  const [accA, setAccA] = useState(0.85);
  const [accB, setAccB] = useState(0.65);
  const [temperature, setTemperature] = useState(0.7);
  const poll = useRef<number | null>(null);
  const api = conn ? new LiveApi(conn) : null;

  const refresh = useCallback(async (a: LiveApi) => {
    setRuns(await a.runs());
  }, []);

  useEffect(() => {
    if (!conn) return;
    const a = new LiveApi(conn);
    setStatus("Connecting… (a free Render instance can take about a minute to wake up)");
    a.health()
      .then((h) => {
        setStatus(`Connected to LLMBench ${h.version}`);
        setError(null);
        return refresh(a);
      })
      .catch((e: Error) => {
        setStatus(null);
        setError(e.message === "Failed to fetch" ? "Could not reach the API (check the URL, that it is awake, and that LLMBENCH_CORS_ORIGINS allows this site)." : e.message);
      });
  }, [conn, refresh]);

  useEffect(() => () => {
    if (poll.current) window.clearInterval(poll.current);
  }, []);

  const connect = () => {
    const next = { baseUrl: draft.baseUrl.trim(), apiKey: draft.apiKey.trim() };
    saveConnection(next);
    setConn(next);
  };

  const disconnect = () => {
    saveConnection(null);
    setConn(null);
    setRuns([]);
    setSummary(null);
    setComparison(null);
    setStatus(null);
  };

  const open = async (runId: string) => {
    if (!api) return;
    setComparison(null);
    setError(null);
    try {
      setSummary(await api.results(runId));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const compare = async () => {
    if (!api || picked.length !== 2) return;
    setSummary(null);
    setError(null);
    try {
      setComparison(await api.compare(picked[0]!, picked[1]!));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const launch = async () => {
    if (!api) return;
    setError(null);
    try {
      const submitted = await api.submit(buildConfig(preset, samples, trials, accA, accB, temperature));
      setJob(submitted);
      if (poll.current) window.clearInterval(poll.current);
      poll.current = window.setInterval(async () => {
        try {
          const s = await api.job(submitted.job_id);
          setJob(s);
          if (s.status === "completed" || s.status === "failed") {
            if (poll.current) window.clearInterval(poll.current);
            await refresh(api);
            if (s.status === "completed" && s.run_id) await open(s.run_id);
          }
        } catch (e) {
          setError((e as Error).message);
        }
      }, 1000);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const togglePick = (id: string) =>
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-2)));

  return (
    <>
      <PageHead title="Live API">
        Connect this dashboard to your own LLMBench API (for example the free Render deployment) to launch benchmarks, watch
        them run and compare results. The key is kept only in this browser's local storage.
      </PageHead>

      <Card title="Connection" actions={conn && <button className="btn" onClick={disconnect}>Disconnect</button>}>
        <div className="form-row">
          <label className="field">
            API base URL
            <input placeholder="https://llmbench-api.onrender.com" value={draft.baseUrl} onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })} />
          </label>
          <label className="field">
            API key (X-API-Key)
            <input type="password" placeholder="LLMBENCH_API_KEY" value={draft.apiKey} onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })} />
          </label>
          <div>
            <button className="btn primary" onClick={connect} disabled={!draft.baseUrl || !draft.apiKey}>
              Connect
            </button>
          </div>
        </div>
        {status && <p className="card-sub">{status}</p>}
        {error && <p className="error">{error}</p>}
        {!conn && (
          <p className="card-sub">
            No API yet? Deploy one for free with the <span className="mono">render.yaml</span> blueprint in the repository (see the README's
            "Free deployment" section), then paste its URL and generated key here.
          </p>
        )}
      </Card>

      {conn && (
        <>
          <Card title="Launch a benchmark" sub="Runs on the API's mock provider: offline, deterministic and free.">
            <div className="form-row">
              <label className="field">
                Task
                <select value={preset} onChange={(e) => setPreset(e.target.value as Preset)}>
                  {Object.entries(PRESETS).map(([id, p]) => (
                    <option value={id} key={id}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                Samples: {samples}
                <input type="range" min={5} max={40} value={samples} onChange={(e) => setSamples(Number(e.target.value))} />
              </label>
              <label className="field">
                Trials per sample: {trials}
                <input type="range" min={1} max={10} value={trials} onChange={(e) => setTrials(Number(e.target.value))} />
              </label>
              <label className="field">
                mock-strong accuracy: {accA.toFixed(2)}
                <input type="range" min={0.1} max={0.99} step={0.01} value={accA} onChange={(e) => setAccA(Number(e.target.value))} />
              </label>
              <label className="field">
                mock-fast accuracy: {accB.toFixed(2)}
                <input type="range" min={0.1} max={0.99} step={0.01} value={accB} onChange={(e) => setAccB(Number(e.target.value))} />
              </label>
              <label className="field">
                Temperature: {temperature.toFixed(1)}
                <input type="range" min={0} max={1.5} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
              </label>
            </div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 14, flexWrap: "wrap" }}>
              <button className="btn primary" onClick={launch} disabled={job?.status === "running" || job?.status === "queued"}>
                Run benchmark
              </button>
              <span className="card-sub">
                {2 * samples * trials} trials · 2 models × 1 prompt × {samples} samples × {trials} trials
              </span>
            </div>
            {job && (
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 6 }}>
                <div className="card-sub">
                  Job <span className="mono">{job.job_id.slice(0, 8)}</span>: {job.status}
                  {job.total ? ` · ${job.done}/${job.total} trials` : ""}
                  {job.error ? ` · ${job.error}` : ""}
                </div>
                <div className="progress">
                  <div style={{ width: `${job.total ? (100 * job.done) / job.total : job.status === "completed" ? 100 : 4}%` }} />
                </div>
              </div>
            )}
          </Card>

          <Card
            title="Runs"
            sub="Click a run to open its report; tick two runs to compare them."
            actions={
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn" onClick={() => api && refresh(api)}>
                  Refresh
                </button>
                <button className="btn primary" disabled={picked.length !== 2} onClick={compare}>
                  Compare selected ({picked.length}/2)
                </button>
              </div>
            }
          >
            {runs.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th />
                      <th>Run</th>
                      <th>Status</th>
                      <th>Task</th>
                      <th className="num">Trials</th>
                      <th className="num">Cost</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr key={r.run_id} className="clickable" onClick={() => open(r.run_id)}>
                        <td onClick={(e) => e.stopPropagation()}>
                          <input type="checkbox" checked={picked.includes(r.run_id)} onChange={() => togglePick(r.run_id)} aria-label={`select ${r.run_id}`} />
                        </td>
                        <td className="mono">{r.run_id}</td>
                        <td>
                          <Badge tone={r.status === "completed" ? "good" : r.status === "running" ? "accent" : "warn"}>{r.status}</Badge>
                        </td>
                        <td>{r.task}</td>
                        <td className="num">
                          {r.n_done}/{r.n_expected}
                        </td>
                        <td className="num">{usd(r.total_cost_usd)}</td>
                        <td>{when(r.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">No runs yet. Launch one above.</div>
            )}
          </Card>

          {summary && (
            <>
              <PageHead title={summary.name}>
                Run <span className="mono">{summary.run_id}</span>
              </PageHead>
              <RunView summary={summary} isMock={(summary.manifest.models ?? []).every((m) => m.provider === "mock")} />
            </>
          )}
          {comparison && (
            <>
              <PageHead title="Comparison" />
              <ComparisonPanel comparison={comparison} />
            </>
          )}
        </>
      )}
    </>
  );
}
