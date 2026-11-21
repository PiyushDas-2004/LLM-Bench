"""Lexical and surface-form metrics."""

from __future__ import annotations

import re
from typing import ClassVar

from llmbench.datasets.schema import Sample
from llmbench.domain import TrialRecord
from llmbench.embeddings import tokenize
from llmbench.metrics.base import MetricContext, MetricKind, TrialMetric, register_metric
from llmbench.text import contains_phrase, normalize_answer, rouge_l, strip_citations, token_f1


def _references(sample: Sample) -> list[str]:
    if sample.reference is None:
        return []
    return [str(sample.reference), *map(str, sample.metadata.get("aliases") or [])]


def _answer_text(trial: TrialRecord) -> str:
    parsed = trial.parsed_output
    return parsed if isinstance(parsed, str) else trial.output_text


@register_metric
class ExactMatch(TrialMetric):
    """Normalised exact match against the reference or any alias (1/0)."""

    name = "exact_match"
    description = "Normalised exact match with the reference or an alias."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 on a normalised match, else 0.0; ``None`` without a reference."""
        refs = _references(sample)
        if not refs:
            return None
        answer = normalize_answer(strip_citations(_answer_text(trial)))
        return float(any(answer == normalize_answer(r) for r in refs))


@register_metric
class TokenF1(TrialMetric):
    """SQuAD token F1, maximised over reference and aliases."""

    name = "f1"
    description = "Token-level F1 with the reference (best over aliases)."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Best token F1 over acceptable answers."""
        refs = _references(sample)
        if not refs:
            return None
        answer = strip_citations(_answer_text(trial))
        return max(token_f1(answer, r) for r in refs)


@register_metric
class RougeL(TrialMetric):
    """ROUGE-L F-measure, maximised over reference and aliases."""

    name = "rouge_l"
    description = "ROUGE-L (LCS) F-measure with the reference."

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """Best ROUGE-L over acceptable answers."""
        refs = _references(sample)
        if not refs:
            return None
        answer = strip_citations(_answer_text(trial))
        return max(rouge_l(answer, r) for r in refs)


@register_metric
class Contains(TrialMetric):
    """Whether the reference (or an alias) appears as a phrase in the output."""

    name = "contains"
    description = "Reference phrase contained in the output (1/0)."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 when any acceptable answer is contained."""
        refs = _references(sample)
        if not refs:
            return None
        return float(any(contains_phrase(trial.output_text, r) for r in refs))


@register_metric
class RegexMatch(TrialMetric):
    """Output matches ``params.pattern`` or ``sample.metadata.regex``."""

    name = "regex"
    description = "Output matches a regular expression (1/0)."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 when the pattern is found anywhere in the output."""
        pattern = self.params.get("pattern") or sample.metadata.get("regex")
        if not pattern:
            return None
        return float(re.search(str(pattern), trial.output_text) is not None)


@register_metric
class LengthOk(TrialMetric):
    """Output length within ``[min_chars, max_chars]`` (and optionally ``max_words``)."""

    name = "length_ok"
    description = "Output length within configured bounds (1/0)."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 when all configured bounds hold."""
        text = trial.output_text.strip()
        lo = int(self.params.get("min_chars", 1))
        hi = self.params.get("max_chars", sample.metadata.get("max_chars"))
        max_words = self.params.get("max_words")
        ok = len(text) >= lo and (hi is None or len(text) <= int(hi))
        if max_words is not None:
            ok = ok and len(text.split()) <= int(max_words)
        return float(ok)


_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        [
            "the",
            "and",
            "is",
            "of",
            "to",
            "in",
            "that",
            "it",
            "for",
            "was",
            "on",
            "are",
            "with",
            "as",
            "be",
            "this",
            "have",
        ]
    ),
    "es": frozenset(
        [
            "el",
            "la",
            "de",
            "que",
            "y",
            "en",
            "los",
            "se",
            "del",
            "las",
            "por",
            "un",
            "una",
            "es",
            "con",
            "para",
        ]
    ),
    "fr": frozenset(
        [
            "le",
            "la",
            "les",
            "de",
            "des",
            "et",
            "est",
            "un",
            "une",
            "du",
            "en",
            "que",
            "pour",
            "dans",
            "qui",
            "pas",
        ]
    ),
    "de": frozenset(
        [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "nicht",
            "ein",
            "eine",
            "zu",
            "den",
            "mit",
            "von",
            "sich",
            "des",
            "auf",
        ]
    ),
    "it": frozenset(
        [
            "il",
            "di",
            "che",
            "la",
            "e",
            "un",
            "una",
            "per",
            "non",
            "sono",
            "del",
            "della",
            "con",
            "gli",
            "le",
        ]
    ),
    "pt": frozenset(
        ["o", "a", "de", "que", "e", "do", "da", "em", "um", "uma", "para", "com", "os", "no", "se"]
    ),
}


def detect_language(text: str) -> str | None:
    """Tiny stop-word language guesser (en/es/fr/de/it/pt); ``None`` if undecided."""
    tokens = tokenize(text)
    if not tokens:
        return None
    scores = {lang: sum(t in words for t in tokens) for lang, words in _STOPWORDS.items()}
    best = max(scores, key=lambda lang: scores[lang])
    return best if scores[best] > 0 else None


@register_metric
class LanguageMatch(TrialMetric):
    """Output language equals ``params.expected`` (default ``en``), via a stop-word heuristic."""

    name = "language"
    description = "Output is in the expected language (heuristic, 1/0)."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 if the detected language matches; ``None`` for very short outputs."""
        if len(tokenize(trial.output_text)) < 4:
            return None
        expected = str(self.params.get("expected", sample.metadata.get("language", "en")))
        return float(detect_language(trial.output_text) == expected)


@register_metric
class TaskPass(TrialMetric):
    """The task's own pass criterion (see :mod:`llmbench.tasks`)."""

    name = "pass"
    description = "Task-specific pass/fail criterion."
    kind: ClassVar[MetricKind] = "binary"

    def score(self, trial: TrialRecord, sample: Sample, ctx: MetricContext) -> float | None:
        """1.0 when the trial passed."""
        return float(bool(trial.passed))
