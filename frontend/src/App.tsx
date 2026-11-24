import { useEffect, useState, type ReactNode } from "react";
import type { Calibration, DemoManifest, GateReport, RunComparison, RunSummary, SweepReport } from "./types";
import { demo } from "./data";
import { ErrorBox, Loading, PageHead, TooltipProvider } from "./components/ui";
import { Overview } from "./views/Overview";
import { RunView } from "./views/RunView";
import { ComparisonPanel } from "./views/CompareView";
import { GateView } from "./views/GateView";
import { RagView } from "./views/RagView";
import { CalibrationView } from "./views/CalibrationView";
import { LiveView } from "./views/LiveView";

type Route = "overview" | "run" | "compare" | "gate" | "safety" | "rag" | "calibration" | "live";

const NAV: { group: string; items: { route: Route; href: string; label: string; icon: string }[] }[] = [
  { group: "Explore", items: [{ route: "overview", href: "#/", label: "Overview", icon: "◎" }] },
  {
    group: "Demo results",
    items: [
      { route: "run", href: "#/run/quickstart", label: "Run report", icon: "▤" },
      { route: "compare", href: "#/compare", label: "Comparison", icon: "⇄" },
      { route: "gate", href: "#/gate", label: "Eval gate", icon: "⛉" },
      { route: "safety", href: "#/safety", label: "Safety suite", icon: "⚑" },
      { route: "rag", href: "#/rag", label: "RAG sweep", icon: "⌕" },
      { route: "calibration", href: "#/calibration", label: "Judge calibration", icon: "⚖" },
    ],
  },
  { group: "Your deployment", items: [{ route: "live", href: "#/live", label: "Live API", icon: "⚡" }] },
];

function parseRoute(hash: string): Route {
  const path = hash.replace(/^#\/?/, "").split("/")[0] ?? "";
  const known: Route[] = ["run", "compare", "gate", "safety", "rag", "calibration", "live"];
  return (known as string[]).includes(path) ? (path as Route) : "overview";
}

/** Load a demo JSON file once and keep it for the session. */
function useDemo<T>(loader: () => Promise<T>, enabled: boolean): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled || data) return;
    loader()
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, [enabled, data, loader]);
  return { data, error };
}

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return window.localStorage.getItem("llmbench.theme") ?? "system";
    } catch {
      return "system";
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem("llmbench.theme", theme);
    } catch {
      // storage unavailable; the choice lasts for this page only
    }
  }, [theme]);
  const next = () => setTheme((t) => (t === "system" ? "dark" : t === "dark" ? "light" : "system"));
  return [theme, next];
}

const loaders = {
  manifest: demo.manifest,
  quickstart: () => demo.run("quickstart"),
  safety: () => demo.run("safety"),
  comparison: demo.comparison,
  gate: demo.gate,
  sweep: demo.sweep,
  calibration: demo.calibration,
};

export function App() {
  const [route, setRoute] = useState<Route>(parseRoute(window.location.hash));
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, toggleTheme] = useTheme();
  useEffect(() => {
    const onHash = () => {
      setRoute(parseRoute(window.location.hash));
      setMenuOpen(false);
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const manifest = useDemo<DemoManifest>(loaders.manifest, true);
  const quickstart = useDemo<RunSummary>(loaders.quickstart, route === "overview" || route === "run");
  const gate = useDemo<GateReport>(loaders.gate, route === "overview" || route === "gate");
  const safety = useDemo<RunSummary>(loaders.safety, route === "safety");
  const comparison = useDemo<RunComparison>(loaders.comparison, route === "compare");
  const sweep = useDemo<SweepReport>(loaders.sweep, route === "rag");
  const calibration = useDemo<Calibration>(loaders.calibration, route === "calibration");

  const body = (): ReactNode => {
    const show = <T,>(state: { data: T | null; error: string | null }, what: string, render: (d: T) => ReactNode) =>
      state.error ? <ErrorBox error={state.error} /> : state.data ? render(state.data) : <Loading what={what} />;
    switch (route) {
      case "run":
        return (
          <>
            <PageHead title="Run report">
              Quickstart benchmark: two simulated models × two prompt versions × 40 questions × 5 trials, with an LLM judge.
            </PageHead>
            {show(quickstart, "run report", (d) => <RunView summary={d} isMock />)}
          </>
        );
      case "compare":
        return (
          <>
            <PageHead title="Statistical comparison">
              Is the difference real? Paired tests on aligned samples, Holm-corrected, with effect sizes and the minimum
              detectable effect.
            </PageHead>
            {show(comparison, "comparison", (d) => <ComparisonPanel comparison={d} />)}
          </>
        );
      case "gate":
        return (
          <>
            <PageHead title="CI eval gate">What a pull request sees when a change makes the model worse.</PageHead>
            {show(gate, "gate report", (d) => <GateView report={d} />)}
          </>
        );
      case "safety":
        return (
          <>
            <PageHead title="Safety suite">
              Prompt-injection, system-prompt leakage and exfiltration probes (harmless canary strings) plus benign controls for
              over-refusal. A "hardened" and a "naive" simulated model.
            </PageHead>
            {show(safety, "safety report", (d) => <RunView summary={d} isMock />)}
          </>
        );
      case "rag":
        return (
          <>
            <PageHead title="RAG configuration sweep">
              Chunking × retrieval × top-k × reranking on the bundled rag_mini corpus: retrieval metrics against gold evidence,
              faithfulness, citations and abstention.
            </PageHead>
            {show(sweep, "RAG sweep", (d) => <RagView report={d} />)}
          </>
        );
      case "calibration":
        return (
          <>
            <PageHead title="Judge calibration">Can you trust the LLM judge? Agreement with 40 human-labelled answers.</PageHead>
            {show(calibration, "calibration", (d) => <CalibrationView report={d} />)}
          </>
        );
      case "live":
        return <LiveView />;
      default:
        return <Overview manifest={manifest.data} quickstart={quickstart.data} gate={gate.data} />;
    }
  };

  return (
    <TooltipProvider>
      <div className="shell">
        <aside className={`sidebar ${menuOpen ? "open" : ""}`} aria-label="Navigation">
          <a className="brand" href="#/" style={{ color: "inherit", textDecoration: "none" }}>
            <span className="brand-mark" aria-hidden>
              <svg width="18" height="18" viewBox="0 0 32 32">
                <path d="M8 22V10m5 12v-7m5 7V7m5 15v-4" stroke="white" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </span>
            LLMBench
          </a>
          {NAV.map((g) => (
            <nav className="nav-group" key={g.group}>
              <div className="nav-title">{g.group}</div>
              {g.items.map((it) => (
                <a key={it.route} href={it.href} className={`nav-link ${route === it.route ? "active" : ""}`} aria-current={route === it.route ? "page" : undefined}>
                  <span aria-hidden style={{ width: 16, textAlign: "center" }}>
                    {it.icon}
                  </span>
                  {it.label}
                </a>
              ))}
            </nav>
          ))}
          <div className="sidebar-foot">
            <a href="../">Documentation</a>
            <a href="https://github.com/PiyushDas-2004/LLM-Bench" target="_blank" rel="noreferrer">
              GitHub repository
            </a>
            <span>MIT licensed · $0 to run</span>
          </div>
        </aside>
        <div className="main">
          <div className="topbar">
            <button className="icon-btn menu-btn" onClick={() => setMenuOpen((o) => !o)} aria-label="Open navigation">
              ☰
            </button>
            <span className="card-sub">{route === "live" ? "Connected to your API" : "Demo data · mock provider · regenerated on every push"}</span>
            <span className="spacer" />
            <button className="icon-btn" onClick={toggleTheme} aria-label={`Theme: ${theme}`} title={`Theme: ${theme}`}>
              {theme === "dark" ? "☾ Dark" : theme === "light" ? "☀ Light" : "◐ Auto"}
            </button>
          </div>
          <main className="content">{body()}</main>
        </div>
      </div>
    </TooltipProvider>
  );
}
