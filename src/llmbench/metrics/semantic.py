"""Embedding-based semantic metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from llmbench.domain import MetricValue, TrialRecord
from llmbench.embeddings import Embedder, make_embedder
from llmbench.errors import ConfigError
from llmbench.metrics.base import Metric, MetricContext, MetricResult, register_metric
from llmbench.text import strip_citations


def embedder_for(metric: Metric, ctx: MetricContext) -> Embedder:
    """The metric's configured embedder (``params.embedder``) or the context default."""
    spec = metric.params.get("embedder")
    return make_embedder(str(spec)) if spec else ctx.embedder


@register_metric
class SemanticSimilarity(Metric):
    """Cosine similarity between output and reference embeddings.

    With the default offline hash embedder this measures lexical overlap; set
    ``params.embedder: st:<model>`` (``embeddings`` extra) for real semantics.
    """

    name = "semantic_similarity"
    description = "Cosine similarity of output and reference embeddings."

    async def compute(self, trials: Sequence[TrialRecord], ctx: MetricContext) -> MetricResult:
        """Embed outputs and references in two batches and compare row-wise."""
        embedder = embedder_for(self, ctx)
        values: list[float | None] = []
        pending: list[tuple[int, str, str]] = []
        for index, trial in enumerate(trials):
            sample = ctx.samples.get(trial.sample_id)
            if not trial.ok:
                values.append(0.0 if sample is not None else None)
            elif sample is None or sample.reference is None:
                values.append(None)
            else:
                values.append(None)
                pending.append((index, strip_citations(trial.output_text), str(sample.reference)))
        if pending:
            outs = await embedder.embed([p[1] for p in pending])
            refs = await embedder.embed([p[2] for p in pending])
            sims = np.clip(np.sum(outs * refs, axis=1), -1.0, 1.0)
            for (index, _, _), sim in zip(pending, sims, strict=True):
                values[index] = float(sim)
        return self.result(
            [_value(self.name, t, v) for t, v in zip(trials, values, strict=True)],
            embedder=embedder.name,
        )


def _value(name: str, trial: TrialRecord, value: float | None) -> MetricValue:
    return MetricValue(
        metric=name,
        model_id=trial.model_id,
        prompt_id=trial.prompt_id,
        sample_id=trial.sample_id,
        trial_index=trial.trial_index,
        value=value,
    )


@register_metric
class BertScore(Metric):
    """BERTScore F1 (requires ``pip install bert-score``; downloads a model on first use)."""

    name = "bertscore"
    description = "BERTScore F1 against the reference (optional dependency)."

    async def compute(self, trials: Sequence[TrialRecord], ctx: MetricContext) -> MetricResult:
        """Score all trials in one batch with ``bert_score.score``."""
        try:
            from bert_score import score as bert_score
        except ImportError as exc:
            raise ConfigError("metric 'bertscore' needs: pip install bert-score") from exc
        scored = [  # pragma: no cover - requires the optional heavy dependency
            t for t in trials if t.ok and ctx.samples[t.sample_id].reference is not None
        ]
        cands = [t.output_text for t in scored]  # pragma: no cover
        refs = [str(ctx.samples[t.sample_id].reference) for t in scored]  # pragma: no cover
        lang: Any = self.params.get("lang", "en")  # pragma: no cover
        _, _, f1 = bert_score(cands, refs, lang=lang)  # pragma: no cover
        return self.result(  # pragma: no cover
            [_value(self.name, t, float(v)) for t, v in zip(scored, f1.tolist(), strict=True)]
        )
