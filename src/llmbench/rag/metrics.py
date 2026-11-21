"""RAG metrics: retrieval quality against gold evidence, and generation groundedness.

Retrieval relevance is judged against gold evidence passages
(``sample.metadata.evidence[*].text``): a chunk is relevant to a passage when
the passage (or most of its content words) is contained in the chunk.
Generation faithfulness decomposes the answer into atomic claims (sentences)
and verifies each against the retrieved context with a pluggable verifier.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar, Protocol

from llmbench.datasets.schema import Sample
from llmbench.domain import TrialRecord
from llmbench.metrics.base import MetricContext, MetricKind, TrialMetric, register_metric
from llmbench.rag.bm25 import bm25_tokens
from llmbench.rag.chunking import split_sentences
from llmbench.rag.retrieval import RetrievedChunk
from llmbench.tasks.base import is_abstention
from llmbench.text import CITATION_RE, normalize_answer, strip_citations

#: Share of an evidence passage's content tokens a chunk must contain to count as relevant.
EVIDENCE_COVERAGE = 0.6


def _evidence(sample: Sample) -> list[str]:
    return [str(e.get("text", "")) for e in sample.metadata.get("evidence") or [] if e.get("text")]


def chunk_supports_evidence(chunk_text: str, evidence: str) -> bool:
    """True when the chunk contains the evidence verbatim or most of its content tokens."""
    norm = normalize_answer(evidence)
    if not norm:
        return False
    if norm in normalize_answer(chunk_text):
        return True
    ev = set(bm25_tokens(evidence))
    if not ev:
        return False
    return len(ev & set(bm25_tokens(chunk_text))) / len(ev) >= EVIDENCE_COVERAGE


def relevance_vector(retrieved: Sequence[RetrievedChunk], sample: Sample) -> list[bool]:
    """Per retrieved chunk: is it relevant to any gold evidence passage?"""
    evidence = _evidence(sample)
    return [any(chunk_supports_evidence(r.chunk.text, e) for e in evidence) for r in retrieved]


def context_precision(retrieved: Sequence[RetrievedChunk], sample: Sample) -> float | None:
    """Relevant retrieved chunks / retrieved chunks (``None`` without evidence)."""
    if not _evidence(sample) or not retrieved:
        return None
    rel = relevance_vector(retrieved, sample)
    return sum(rel) / len(rel)


def context_recall(retrieved: Sequence[RetrievedChunk], sample: Sample) -> float | None:
    """Gold evidence passages covered by at least one retrieved chunk / gold passages."""
    evidence = _evidence(sample)
    if not evidence:
        return None
    covered = sum(
        any(chunk_supports_evidence(r.chunk.text, e) for r in retrieved) for e in evidence
    )
    return covered / len(evidence)


def hit_at_k(retrieved: Sequence[RetrievedChunk], sample: Sample) -> float | None:
    """1 if any retrieved chunk is relevant."""
    if not _evidence(sample):
        return None
    return float(any(relevance_vector(retrieved, sample)))


def mrr(retrieved: Sequence[RetrievedChunk], sample: Sample) -> float | None:
    """Reciprocal rank of the first relevant chunk (0 if none)."""
    if not _evidence(sample):
        return None
    for rank, relevant in enumerate(relevance_vector(retrieved, sample), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[RetrievedChunk], sample: Sample) -> float | None:
    """Binary-relevance nDCG@k; the ideal ranking places min(k, #evidence) relevant chunks first."""
    evidence = _evidence(sample)
    if not evidence or not retrieved:
        return None
    rel = relevance_vector(retrieved, sample)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), len(evidence))))
    return dcg / ideal if ideal else 0.0


class ClaimVerifier(Protocol):
    """Decides whether a claim is supported by a piece of context."""

    async def supported(self, claim: str, context: str) -> bool:
        """True when ``context`` supports ``claim``."""
        ...


class LexicalVerifier:
    """Offline proxy for NLI: a claim is supported when most of its content words occur in
    the context. Weak by design (misses paraphrase, can't see negation); use
    :class:`JudgeVerifier` for real evaluations.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    async def supported(self, claim: str, context: str) -> bool:
        """Content-token coverage >= threshold."""
        tokens = set(bm25_tokens(claim))
        if not tokens:
            return True
        return len(tokens & set(bm25_tokens(context))) / len(tokens) >= self.threshold


class JudgeVerifier:
    """Verifier backed by an :class:`~llmbench.judges.judge.LLMJudge` (pass = supported)."""

    def __init__(self, judge: object) -> None:
        self.judge = judge

    async def supported(self, claim: str, context: str) -> bool:
        """Ask the judge whether the context supports the claim."""
        verdict = await self.judge.pointwise(  # type: ignore[attr-defined]
            f"Is this claim fully supported by the reference context? Claim: {claim}",
            claim,
            context,
            subject="faithfulness",
        )
        return bool(verdict.passed)


def answer_claims(answer: str) -> list[str]:
    """Atomic claims = sentences of the answer, citations removed, abstentions dropped."""
    if is_abstention(answer):
        return []
    return [s for s in (strip_citations(x) for x in split_sentences(answer)) if bm25_tokens(s)]


async def faithfulness(
    answer: str, retrieved: Sequence[RetrievedChunk], verifier: ClaimVerifier
) -> float | None:
    """Supported claims / claims (``None`` when the answer makes no claims)."""
    claims = answer_claims(answer)
    if not claims:
        return None
    context = "\n".join(r.chunk.text for r in retrieved)
    supported = [await verifier.supported(c, context) for c in claims]
    return sum(supported) / len(supported)


async def citation_accuracy(
    answer: str, retrieved: Sequence[RetrievedChunk], verifier: ClaimVerifier
) -> float | None:
    """Share of ``[n]`` citations whose passage supports the sentence that cites it."""
    checks: list[bool] = []
    for sentence in split_sentences(answer):
        claim = strip_citations(sentence)
        for match in CITATION_RE.findall(sentence):
            n = int(match)
            if not 1 <= n <= len(retrieved):
                checks.append(False)
                continue
            checks.append(await verifier.supported(claim, retrieved[n - 1].chunk.text))
    return sum(checks) / len(checks) if checks else None


def abstention_correct(answer: str, sample: Sample) -> float:
    """1 if the model abstained exactly when the question is unanswerable."""
    answerable = bool(sample.metadata.get("answerable", True))
    return float(is_abstention(answer) != answerable)


@register_metric
class AbstentionCorrect(TrialMetric):
    """Abstained on unanswerable questions and answered answerable ones (1/0)."""

    name = "abstention_correct"
    description = "Correct abstention behaviour on answerable/unanswerable questions."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """See :func:`abstention_correct`."""
        return abstention_correct(trial.output_text, sample)
