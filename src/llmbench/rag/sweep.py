"""RAG evaluation and configuration sweeps.

``run_sweep`` evaluates every combination of chunking, retrieval, top-k,
reranking and embedder settings on a question set, reports each metric with a
bootstrap CI, and computes *main effects*: for every factor, the average of
each metric at each level, so you can see which setting moves which metric.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from llmbench.config.models import JudgeSpec, ModelSpec
from llmbench.config.settings import Settings, get_settings
from llmbench.datasets.loader import load_samples
from llmbench.datasets.schema import Sample
from llmbench.embeddings import Embedder, make_embedder
from llmbench.errors import ConfigError
from llmbench.judges.judge import LLMJudge
from llmbench.judges.rubric import Rubric
from llmbench.prompts.template import PromptTemplate
from llmbench.providers.base import GenerationParams, Provider
from llmbench.providers.registry import check_availability, create_provider
from llmbench.rag import metrics as rm
from llmbench.rag.chunking import ChunkingConfig, chunk_corpus
from llmbench.rag.documents import Document, load_corpus
from llmbench.rag.pipeline import RAGAnswer, RAGPipeline
from llmbench.rag.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LLMReranker,
    RerankingRetriever,
    Retriever,
    VectorRetriever,
)
from llmbench.stats.bootstrap import ConfidenceInterval, percentile_ci
from llmbench.tasks.base import get_task, is_abstention
from llmbench.text import strip_citations, token_f1

RAG_METRICS: tuple[str, ...] = (
    "context_precision",
    "context_recall",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "hallucination",
    "answer_relevance",
    "citation_accuracy",
    "abstention_correct",
    "answer_f1",
    "pass",
)
#: Metrics where lower is better (used to pick the best setting).
LOWER_IS_BETTER = frozenset({"hallucination"})


class SweepGrid(BaseModel):
    """Levels for each factor; the sweep evaluates their Cartesian product."""

    model_config = ConfigDict(extra="forbid")

    chunk_strategy: list[Literal["fixed", "sentence", "recursive"]] = ["recursive"]
    chunk_size: list[int] = [400]
    chunk_overlap: list[int] = [50]
    top_k: list[int] = [4]
    retrieval: list[Literal["bm25", "vector", "hybrid"]] = ["hybrid"]
    reranker: list[Literal["none", "llm"]] = ["none"]
    embedder: list[str] = ["hash"]


class RAGSetting(BaseModel):
    """One point of the sweep grid."""

    model_config = ConfigDict(frozen=True)

    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval: str
    reranker: str
    embedder: str

    @property
    def label(self) -> str:
        """Compact label, e.g. ``recursive/400/50 hybrid k=4 rerank=none emb=hash``."""
        return (
            f"{self.chunk_strategy}/{self.chunk_size}/{self.chunk_overlap} {self.retrieval} "
            f"k={self.top_k} rerank={self.reranker} emb={self.embedder}"
        )


class RAGSweepConfig(BaseModel):
    """YAML schema for ``llmbench rag sweep``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    corpus: str
    questions: str
    prompt: str
    generator: ModelSpec
    verifier: Literal["lexical", "judge"] = "lexical"
    judge: JudgeSpec | None = None
    sweep: SweepGrid = Field(default_factory=SweepGrid)
    n_resamples: int = Field(default=1000, ge=100)
    seed: int = 0

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGSweepConfig:
        """Load a sweep config; relative paths resolve against the file, then the cwd."""
        file = Path(path)
        if not file.is_file():
            raise ConfigError(f"file not found: {file}")
        config = cls.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))
        config._base_dir = file.parent.resolve()
        return config

    def resolve(self, value: str) -> Path:
        """Resolve a path from the config."""
        path = Path(value)
        if path.is_absolute():
            return path
        for root in (self._base_dir, Path.cwd()):
            if (root / path).exists():
                return (root / path).resolve()
        return (self._base_dir / path).resolve()

    def settings(self) -> list[RAGSetting]:
        """All grid points with valid chunking (overlap < size)."""
        grid = self.sweep
        combos = itertools.product(
            grid.chunk_strategy,
            grid.chunk_size,
            grid.chunk_overlap,
            grid.top_k,
            grid.retrieval,
            grid.reranker,
            grid.embedder,
        )
        return [
            RAGSetting(
                chunk_strategy=s,
                chunk_size=size,
                chunk_overlap=ov,
                top_k=k,
                retrieval=r,
                reranker=rr,
                embedder=e,
            )
            for s, size, ov, k, r, rr, e in combos
            if ov < size
        ]


class QuestionResult(BaseModel):
    """Per-question outputs and metric values."""

    sample_id: str
    answer: str
    retrieved_ids: list[str]
    metrics: dict[str, float | None]


class SettingResult(BaseModel):
    """Aggregates for one setting."""

    setting: RAGSetting
    n_questions: int
    metrics: dict[str, ConfidenceInterval | None]
    mean_latency_ms: float
    questions: list[QuestionResult]

    def value(self, metric: str) -> float | None:
        """Point estimate of a metric."""
        ci = self.metrics.get(metric)
        return ci.estimate if ci else None


class SweepReport(BaseModel):
    """Full sweep output."""

    name: str
    generator: str
    verifier: str
    n_questions: int
    results: list[SettingResult]
    main_effects: dict[str, dict[str, dict[str, float]]]
    best: dict[str, str]


async def score_answer(
    answer: RAGAnswer, sample: Sample, verifier: rm.ClaimVerifier, embedder: Embedder
) -> dict[str, float | None]:
    """All RAG metrics for one answered question."""
    retrieved = answer.retrieved
    faith = await rm.faithfulness(answer.answer, retrieved, verifier)
    answerable = bool(sample.metadata.get("answerable", True))
    abstained = is_abstention(answer.answer)
    relevance: float | None = None
    if answer.answer and not abstained:
        vectors = await embedder.embed([sample.input, strip_citations(answer.answer)])
        relevance = float(np.clip(vectors[0] @ vectors[1], -1, 1))
    reference = str(sample.reference) if sample.reference is not None else ""
    return {
        "context_precision": rm.context_precision(retrieved, sample),
        "context_recall": rm.context_recall(retrieved, sample),
        "hit_at_k": rm.hit_at_k(retrieved, sample),
        "mrr": rm.mrr(retrieved, sample),
        "ndcg_at_k": rm.ndcg_at_k(retrieved, sample),
        "faithfulness": faith,
        "hallucination": None if faith is None else float(faith < 1.0),
        "answer_relevance": relevance,
        "citation_accuracy": await rm.citation_accuracy(answer.answer, retrieved, verifier),
        "abstention_correct": rm.abstention_correct(answer.answer, sample),
        "answer_f1": token_f1(strip_citations(answer.answer), reference) if answerable else None,
        "pass": float(get_task("rag").is_pass(answer.answer, sample)),
    }


async def build_retriever(
    setting: RAGSetting, docs: list[Document], reranker_provider: tuple[Provider, str] | None
) -> Retriever:
    """Construct the retriever stack for a setting."""
    chunks = chunk_corpus(
        docs,
        ChunkingConfig.model_validate(
            {
                "strategy": setting.chunk_strategy,
                "size": setting.chunk_size,
                "overlap": setting.chunk_overlap,
            }
        ),
    )
    lexical = BM25Retriever(chunks)
    retriever: Retriever = lexical
    if setting.retrieval in {"vector", "hybrid"}:
        dense = VectorRetriever(make_embedder(setting.embedder))
        await dense.index(chunks)
        retriever = dense if setting.retrieval == "vector" else HybridRetriever(lexical, dense)
    if setting.reranker == "llm":
        if reranker_provider is None:
            raise ConfigError("reranker 'llm' requires a generator provider")
        retriever = RerankingRetriever(retriever, LLMReranker(*reranker_provider), candidates=10)
    return retriever


def _ci(values: list[float | None], n_resamples: int, seed: int) -> ConfidenceInterval | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    if len(clean) == 1:
        v = clean[0]
        return ConfidenceInterval(
            estimate=v, low=v, high=v, level=0.95, method="single", n_samples=1, n_trials=1
        )
    return percentile_ci(clean, n_resamples=n_resamples, seed=seed)


def main_effects(results: list[SettingResult]) -> dict[str, dict[str, dict[str, float]]]:
    """factor -> metric -> level -> mean of setting estimates (factors with >1 level only)."""
    factors = list(RAGSetting.model_fields)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for factor in factors:
        levels = {str(getattr(r.setting, factor)) for r in results}
        if len(levels) < 2:
            continue
        out[factor] = {}
        for metric in RAG_METRICS:
            by_level: dict[str, list[float]] = defaultdict(list)
            for r in results:
                value = r.value(metric)
                if value is not None:
                    by_level[str(getattr(r.setting, factor))].append(value)
            if by_level:
                out[factor][metric] = {
                    lvl: float(np.mean(v)) for lvl, v in sorted(by_level.items())
                }
    return out


def _make_verifier(config: RAGSweepConfig, settings: Settings) -> rm.ClaimVerifier:
    if config.verifier == "lexical":
        return rm.LexicalVerifier()
    if config.judge is None:
        raise ConfigError("verifier 'judge' requires a 'judge' section")
    spec = ModelSpec(
        id="judge",
        provider=config.judge.provider,
        model=config.judge.model,
        base_url=config.judge.base_url,
        params=dict(config.judge.params),
    )
    availability = check_availability(spec, settings)
    if not availability.ok:
        raise ConfigError(f"judge cannot run: {availability.reason}")
    judge = LLMJudge(
        provider=create_provider(spec, settings),
        provider_name=spec.provider,
        model=spec.model_name,
        rubric=Rubric.from_yaml(config.resolve(config.judge.rubric)),
        params=config.judge.params,
    )
    return rm.JudgeVerifier(judge)


async def run_sweep(config: RAGSweepConfig, settings: Settings | None = None) -> SweepReport:
    """Evaluate every setting in the grid."""
    settings = settings or get_settings()
    availability = check_availability(config.generator, settings)
    if not availability.ok:
        raise ConfigError(f"generator cannot run: {availability.reason}")
    docs = load_corpus(config.resolve(config.corpus))
    samples = load_samples(config.resolve(config.questions))
    prompt_path = config.resolve(config.prompt)
    if not prompt_path.is_file():
        raise ConfigError(f"prompt template not found: {prompt_path}")
    prompt = PromptTemplate("rag", prompt_path.read_text(encoding="utf-8"))
    provider = create_provider(config.generator, settings)
    model = config.generator.model_name
    verifier = _make_verifier(config, settings)
    params = GenerationParams.from_mapping(config.generator.params)
    results: list[SettingResult] = []
    try:
        for index, setting in enumerate(config.settings()):
            retriever = await build_retriever(setting, docs, (provider, model))
            pipeline = RAGPipeline(
                retriever, provider, model, prompt, top_k=setting.top_k, params=params
            )
            embedder = make_embedder(setting.embedder)
            questions: list[QuestionResult] = []
            latencies: list[float] = []
            for sample in samples:
                answer = await pipeline.answer(sample)
                latencies.append(answer.latency_ms)
                questions.append(
                    QuestionResult(
                        sample_id=sample.id,
                        answer=answer.answer,
                        retrieved_ids=[r.chunk.id for r in answer.retrieved],
                        metrics=await score_answer(answer, sample, verifier, embedder),
                    )
                )
            metrics = {
                m: _ci([q.metrics[m] for q in questions], config.n_resamples, config.seed + index)
                for m in RAG_METRICS
            }
            results.append(
                SettingResult(
                    setting=setting,
                    n_questions=len(samples),
                    metrics=metrics,
                    mean_latency_ms=float(np.mean(latencies)),
                    questions=questions,
                )
            )
    finally:
        await provider.aclose()
    best: dict[str, str] = {}
    for metric in RAG_METRICS:
        scored = [
            (r.value(metric), r.setting.label) for r in results if r.value(metric) is not None
        ]
        if scored:
            pick: Any = min if metric in LOWER_IS_BETTER else max
            best[metric] = pick(scored, key=lambda x: x[0])[1]
    return SweepReport(
        name=config.name,
        generator=f"{config.generator.provider}/{model}",
        verifier=config.verifier,
        n_questions=len(samples),
        results=results,
        main_effects=main_effects(results),
        best=best,
    )


def render_sweep_markdown(report: SweepReport, top: int = 10, sort_by: str = "pass") -> str:
    """Markdown summary: leaderboard, main effects and best setting per metric."""
    lines = [
        f"# RAG sweep: {report.name}",
        "",
        f"Generator `{report.generator}`, verifier `{report.verifier}`, "
        f"{report.n_questions} questions, {len(report.results)} settings.",
        "",
        f"## Top {min(top, len(report.results))} settings by `{sort_by}`",
        "",
    ]
    cols = [
        "pass",
        "answer_f1",
        "context_recall",
        "mrr",
        "ndcg_at_k",
        "faithfulness",
        "abstention_correct",
    ]
    lines.append("| setting | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    ranked = sorted(report.results, key=lambda r: -(r.value(sort_by) or -1))
    for r in ranked[:top]:
        cells = []
        for c in cols:
            ci = r.metrics.get(c)
            cells.append(f"{ci.estimate:.3f} [{ci.low:.2f}, {ci.high:.2f}]" if ci else "n/a")
        lines.append(f"| `{r.setting.label}` | " + " | ".join(cells) + " |")
    lines += ["", "## Main effects (mean metric at each factor level)", ""]
    for factor, per_metric in report.main_effects.items():
        lines.append(f"### {factor}")
        lines.append("")
        lines.append(
            "| metric | " + " | ".join(next(iter(per_metric.values())).keys()) + " | spread |"
        )
        lines.append("|---|" + "---|" * (len(next(iter(per_metric.values()))) + 1))
        for metric, levels in per_metric.items():
            spread = max(levels.values()) - min(levels.values())
            lines.append(
                f"| {metric} | "
                + " | ".join(f"{v:.3f}" for v in levels.values())
                + f" | {spread:.3f} |"
            )
        lines.append("")
    lines += ["## Best setting per metric", ""]
    lines += [f"- **{m}**: `{label}`" for m, label in report.best.items()]
    return "\n".join(lines) + "\n"
