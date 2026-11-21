"""Consistency metrics: how stable are answers across repeated trials of the same input?

All of them operate on the trials of one sample within one configuration.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from itertools import combinations
from typing import Any, ClassVar

import numpy as np

from llmbench.datasets.schema import Sample
from llmbench.domain import TrialRecord
from llmbench.metrics.base import GroupMetric, MetricContext, MetricKind, register_metric
from llmbench.metrics.semantic import embedder_for
from llmbench.text import normalize_answer, strip_citations


def canonical_answer(trial: TrialRecord) -> str:
    """Normalised answer used to compare trials (JSON canonicalised, text normalised)."""
    if not trial.ok:
        return f"<error:{trial.error_class}>"
    parsed: Any = trial.parsed_output
    if isinstance(parsed, (dict, list)):
        return json.dumps(parsed, sort_keys=True)
    text = parsed if isinstance(parsed, str) else trial.output_text
    return normalize_answer(strip_citations(text))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k = 1 - C(n-c, k) / C(n, k): P(at least one of k draws passes)."""
    if not 0 < k <= n:
        raise ValueError(f"need 0 < k <= n (k={k}, n={n})")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k = C(c, k) / C(n, k): P(all of k draws pass)."""
    if not 0 < k <= n:
        raise ValueError(f"need 0 < k <= n (k={k}, n={n})")
    return math.comb(c, k) / math.comb(n, k)


def _k(metric: GroupMetric, n: int) -> int:
    k = int(metric.params.get("k", min(3, n)))
    return max(1, min(k, n))


@register_metric
class ExactAgreement(GroupMetric):
    """Fraction of trials that equal the modal (most common) normalised answer."""

    name = "exact_agreement"
    description = "Share of trials matching the modal answer (1.0 = fully consistent)."

    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """Modal count / number of trials."""
        counts = Counter(canonical_answer(t) for t in trials)
        return counts.most_common(1)[0][1] / len(trials)


@register_metric
class AnswerEntropy(GroupMetric):
    """Shannon entropy (bits) of the distribution of normalised answers; 0 = deterministic."""

    name = "answer_entropy"
    description = "Entropy in bits of normalised answers across trials (lower = more stable)."
    higher_is_better = False

    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """-sum p log2 p over distinct answers."""
        counts = np.array(list(Counter(canonical_answer(t) for t in trials).values()), float)
        p = counts / counts.sum()
        return float(-(p * np.log2(p)).sum()) + 0.0


@register_metric
class SemanticConsistency(GroupMetric):
    """Mean pairwise cosine similarity between the outputs of different trials."""

    name = "semantic_consistency"
    description = "Mean pairwise embedding similarity across trials."

    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """Average of the upper triangle of the trial-by-trial similarity matrix."""
        ok = [t for t in trials if t.ok]
        if len(ok) < 2:
            return None
        vectors = await embedder_for(self, ctx).embed([strip_citations(t.output_text) for t in ok])
        sims = [float(vectors[i] @ vectors[j]) for i, j in combinations(range(len(ok)), 2)]
        return float(np.clip(np.mean(sims), -1.0, 1.0))


@register_metric
class PassAtK(GroupMetric):
    """Unbiased pass@k across trials (``params.k``, default min(3, trials))."""

    name = "pass_at_k"
    description = "Probability that at least one of k sampled trials passes."

    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """pass@k from n trials with c passes."""
        n, c = len(trials), sum(bool(t.passed) for t in trials)
        return pass_at_k(n, c, _k(self, n))


@register_metric
class PassHatK(GroupMetric):
    """Unbiased pass^k across trials: probability that *all* k sampled trials pass."""

    name = "pass_hat_k"
    description = "Probability that all of k sampled trials pass (reliability)."

    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """pass^k from n trials with c passes."""
        n, c = len(trials), sum(bool(t.passed) for t in trials)
        return pass_hat_k(n, c, _k(self, n))


@register_metric
class FlipRate(GroupMetric):
    """1 if the item both passes and fails across trials (an unstable item), else 0."""

    name = "flip_rate"
    description = "Share of items whose pass/fail outcome flips across trials."
    higher_is_better = False
    kind: ClassVar[MetricKind] = "binary"

    async def score_group(
        self, trials: list[TrialRecord], sample: Sample, ctx: MetricContext
    ) -> float | None:
        """``None`` with a single trial (flips are undefined)."""
        if len(trials) < 2:
            return None
        outcomes = {bool(t.passed) for t in trials}
        return float(len(outcomes) == 2)
