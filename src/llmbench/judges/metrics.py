"""Judge-backed metrics registered in the metric registry."""

from __future__ import annotations

from collections.abc import Sequence

from scipy import stats as sps

from llmbench.domain import MetricValue, TrialRecord
from llmbench.judges.judge import LLMJudge
from llmbench.metrics.base import Metric, MetricContext, MetricResult, register_metric
from llmbench.text import strip_citations


def length_bias(lengths: Sequence[int], scores: Sequence[float]) -> float | None:
    """Spearman correlation between answer length and judge score (``None`` if undefined)."""
    if len(scores) < 3 or len(set(scores)) < 2 or len(set(lengths)) < 2:
        return None
    rho = sps.spearmanr(lengths, scores).statistic
    return float(rho)


@register_metric
class LLMJudgeCorrectness(Metric):
    """Rubric-based correctness from the configured LLM judge, normalised to [0, 1].

    Result metadata reports the judge's self-consistency (when ``samples`` > 1) and
    the length-bias correlation; each value's details keep the raw score and answer
    length so reports can recompute bias. Judge-equals-candidate is flagged by
    :func:`llmbench.judges.factory.build_judge` (warning) and in run summaries.
    """

    name = "llm_judge_correctness"
    description = "LLM-judge rubric score normalised to [0, 1]."

    async def compute(self, trials: Sequence[TrialRecord], ctx: MetricContext) -> MetricResult:
        """Judge every successful trial; failed calls score 0."""
        judge: LLMJudge | None = ctx.judge
        if judge is None:
            skipped = [
                _value(self.name, t, None, {"skipped": "no judge available"}) for t in trials
            ]
            return self.result(skipped, skipped="judge not configured or unavailable")
        judge.bind(ctx.store, ctx.run_id)
        values: list[MetricValue] = []
        lengths: list[int] = []
        scores: list[float] = []
        consistency: list[float] = []
        for trial in trials:
            sample = ctx.samples.get(trial.sample_id)
            if sample is None:
                continue
            if not trial.ok:
                values.append(_value(self.name, trial, 0.0, {"error": "provider call failed"}))
                continue
            reference = None if sample.reference is None else str(sample.reference)
            verdict = await judge.pointwise(
                sample.input, strip_citations(trial.output_text), reference, subject=trial.trial_key
            )
            values.append(
                _value(
                    self.name,
                    trial,
                    verdict.normalized,
                    {
                        "score": verdict.score,
                        "passed": verdict.passed,
                        "error": verdict.error,
                        "length": len(trial.output_text),
                    },
                )
            )
            if verdict.normalized is not None:
                lengths.append(len(trial.output_text))
                scores.append(verdict.normalized)
            if verdict.self_consistency is not None:
                consistency.append(verdict.self_consistency)
        return self.result(
            values,
            judge=f"{judge.provider_name}/{judge.model}",
            rubric=judge.rubric.name,
            length_bias_spearman=length_bias(lengths, scores),
            judge_self_consistency=sum(consistency) / len(consistency) if consistency else None,
        )


def _value(
    name: str, trial: TrialRecord, value: float | None, details: dict[str, object]
) -> MetricValue:
    return MetricValue(
        metric=name,
        model_id=trial.model_id,
        prompt_id=trial.prompt_id,
        sample_id=trial.sample_id,
        trial_index=trial.trial_index,
        value=value,
        details=details,
    )
