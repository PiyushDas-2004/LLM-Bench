# ADR 0009: Static web dashboard with demo and live modes

- **Status:** Accepted
- **Date:** 2026-11-24

## Context

The HTML report is per run and generated offline. People evaluating LLMBench (and teams
using it) also want to browse results interactively, compare runs and launch benchmarks
without a terminal. The project must stay free to host and free to run.

## Decision

- A single-page dashboard in `frontend/` built with React and TypeScript (strict), bundled by
  Vite. Charts are hand-written SVG components (CI bars, Pareto scatter, histogram, forest
  plot) using CSS colour tokens with light and dark themes, so there is no chart dependency.
- **Demo mode** reads static JSON produced in CI by `scripts/build_site.sh`
  (`llmbench report --format json`, `compare --format json`, `check --json`, `rag sweep --json`,
  judge calibration). It needs no backend and is always on.
- **Live mode** talks to any LLMBench REST API. The browser sends `X-API-Key`; the API allows
  the dashboard origin through `LLMBENCH_CORS_ORIGINS` (off by default). The key is stored only
  in the browser's local storage.
- The build is served from the same GitHub Pages site as the docs, at `/app/`, with a relative
  base path so any static host works.

## Consequences

Positive: zero hosting cost, no server needed for the demo, the same view components render
demo and live data because both use the API's response models, and CI type-checks and builds
the dashboard on every push.

Negative: the TypeScript types mirror the pydantic models by hand and must be kept in sync.
Live mode depends on the API being reachable from the browser (CORS, free-tier wake-up time).
Storing an API key in local storage is acceptable for a personal deployment but not for a
shared, multi-user one, which would need real authentication.

## Alternatives considered

- **Server-rendered pages in FastAPI**: would couple the UI to a running server and cost
  money to keep online.
- **Streamlit / Gradio**: fast to build but needs a Python server for every view and gives
  less control over the visual design.
- **A charting library (Recharts, Plotly)**: larger bundle and less control over accessible,
  theme-aware styling for a handful of chart types.
