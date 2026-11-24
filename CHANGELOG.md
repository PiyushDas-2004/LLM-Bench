# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-21

### Added
- Experiment configuration in YAML (pydantic-validated): models x prompts x samples x
  repeated trials, sampling, rate limits, budgets, retries, circuit breaker, judge, pricing.
- Golden datasets with strict schema, content hashing (dataset versioning) and stratified
  sampling; shipped original datasets for QA, JSON extraction, intent classification,
  RAG (`rag_mini`), prompt-injection safety probes and judge calibration.
- Sandboxed Jinja2 prompt templates with variable validation and content hashes.
- Provider layer with a typed error taxonomy: deterministic mock provider, OpenAI,
  Gemini (google-genai) and OpenAI-compatible servers (Ollama, vLLM, LM Studio).
- Async runner: bounded concurrency, per-provider token buckets (requests and tokens per
  minute), full-jitter retries honouring `Retry-After`, circuit breaker, budget guard,
  per-trial checkpointing with `resume`, SIGINT handling, content-addressed response cache
  keyed by trial index.
- Metric registry: lexical, structural (JSON Schema), semantic, consistency (agreement,
  entropy, unbiased pass@k / pass^k, flip rate), reliability, latency, cost and safety
  metrics; `@trial_metric` decorator and entry-point plugins.
- LLM-as-a-judge with rubrics, JSON validation and repair, self-consistency,
  position-swapped pairwise comparison, length-bias reporting, self-preference warning
  and calibration against human labels (accuracy, Cohen's kappa, confusion matrix).
- RAG module: chunking strategies, BM25, vector and hybrid (RRF) retrieval, LLM
  reranking, cited generation, retrieval metrics (precision, recall, hit@k, MRR, nDCG),
  faithfulness, citation accuracy, abstention correctness and configuration sweeps with
  main-effects analysis; optional pgvector backend.
- Statistics: percentile, BCa and two-stage cluster bootstrap; paired permutation,
  paired bootstrap, McNemar and Wilcoxon tests; Cohen's d / d_z and Cliff's delta;
  Holm and Benjamini-Hochberg corrections; power / minimum detectable effect.
- Run tracking in SQLite (SQLAlchemy 2.0 + Alembic), run manifests, run comparison with
  per-sample regressions and tag breakdowns, statistical regression gates
  (`llmbench check`), drift detection and portable run bundles for CI baselines.
- Reporting: rich CLI tables, self-contained HTML report (Pareto frontier, consistency,
  latency, failures, judge calibration, manifest), Markdown for PR comments, JSON / CSV /
  Parquet exports.
- FastAPI service with an in-process job queue, API-key auth, request IDs, structured
  logging and Prometheus metrics; optional OpenTelemetry spans.
- Docker image (multi-stage, non-root), docker-compose, CI, eval-gate, live-benchmark
  and release workflows.

[Unreleased]: https://github.com/PiyushDas-2004/LLM-Bench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PiyushDas-2004/LLM-Bench/releases/tag/v0.1.0
