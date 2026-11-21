"""LLM-as-a-judge with structured output validation, repair, self-consistency and
position-swapped pairwise comparison. Every call is recorded for auditability.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import structlog
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from llmbench.config.pricing import PricingTable
from llmbench.judges.rubric import Rubric
from llmbench.providers.base import (
    GenerationParams,
    GenerationRequest,
    Message,
    Provider,
    ProviderError,
)
from llmbench.text import extract_json

log = structlog.get_logger(__name__)


class JudgeVerdict(BaseModel):
    """Result of pointwise judging (possibly aggregated over several samples)."""

    score: float | None
    normalized: float | None
    passed: bool | None
    reasoning: str = ""
    sample_scores: list[int] = Field(default_factory=list)
    self_consistency: float | None = None
    attempts: int = 0
    error: str | None = None


class PairwiseVerdict(BaseModel):
    """Result of a position-swapped pairwise comparison."""

    winner: str | None
    first_order: str | None
    swapped_order: str | None
    position_consistent: bool | None
    error: str | None = None


class LLMJudge:
    """Grades answers with an LLM according to a :class:`Rubric`.

    Args:
        provider: Judge provider.
        provider_name: Provider name (for cost lookups and audit records).
        model: Judge model name.
        rubric: Grading rubric.
        params: Generation params (temperature 0 recommended).
        samples: Independent judgments per item; >1 enables self-consistency reporting.
        max_repairs: Re-prompts allowed when the judge returns malformed JSON.
        swap_order: Run pairwise comparisons in both orders and flag disagreement.
        pricing: Pricing table for judge-call cost.
        store: Optional run store; when given every call is persisted.
        run_id: Run the calls belong to (``None`` for calibration).
    """

    def __init__(
        self,
        *,
        provider: Provider,
        provider_name: str,
        model: str,
        rubric: Rubric,
        params: dict[str, Any] | None = None,
        samples: int = 1,
        max_repairs: int = 2,
        swap_order: bool = True,
        pricing: PricingTable | None = None,
        store: Any = None,
        run_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.provider_name = provider_name
        self.model = model
        self.rubric = rubric
        self.params = GenerationParams.from_mapping(params or {"temperature": 0.0})
        self.samples = samples
        self.max_repairs = max_repairs
        self.swap_order = swap_order
        self.pricing = pricing or PricingTable.empty()
        self.store = store
        self.run_id = run_id
        self._point_schema = Draft202012Validator(
            {
                "type": "object",
                "required": ["score"],
                "properties": {
                    "score": {
                        "type": "integer",
                        "minimum": rubric.scale_min,
                        "maximum": rubric.scale_max,
                    },
                    "reasoning": {"type": "string"},
                },
            }
        )
        self._pair_schema = Draft202012Validator(
            {
                "type": "object",
                "required": ["winner"],
                "properties": {
                    "winner": {"enum": ["A", "B", "tie"]},
                    "reasoning": {"type": "string"},
                },
            }
        )

    def bind(self, store: Any, run_id: str | None) -> LLMJudge:
        """Attach an audit store / run id (returns ``self``)."""
        self.store, self.run_id = store, run_id
        return self

    async def _call(
        self,
        system: str,
        user: str,
        validator: Draft202012Validator,
        mock_spec: dict[str, Any],
        *,
        subject: str,
        mode: str,
        sample_index: int,
    ) -> tuple[dict[str, Any] | None, int, str | None]:
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        last_error: str | None = None
        for attempt in range(1, self.max_repairs + 2):
            request = GenerationRequest(
                model=self.model,
                messages=tuple(messages),
                params=self.params.model_copy(update={"json_mode": True}),
                trial_index=sample_index * 100 + attempt,
                metadata={"mock_judge": mock_spec},
            )
            start = time.perf_counter()
            try:
                result = await self.provider.generate(request)
            except ProviderError as exc:
                self._record(subject, mode, user, "", None, 0, 0, None, exc.message)
                return None, attempt, f"{exc.error_class.value}: {exc.message}"
            latency = (time.perf_counter() - start) * 1000
            parsed: Any = None
            try:
                parsed = extract_json(result.text)
                errors = sorted(validator.iter_errors(parsed), key=str)
                last_error = errors[0].message if errors else None
            except ValueError as exc:
                last_error = str(exc)
            cost = self.pricing.cost(
                self.provider_name, self.model, result.input_tokens, result.output_tokens
            )
            self._record(
                subject,
                mode,
                messages[-1].content,
                result.text,
                parsed if not last_error else None,
                result.input_tokens,
                result.output_tokens,
                cost,
                last_error,
                latency,
            )
            if last_error is None and isinstance(parsed, dict):
                return parsed, attempt, None
            messages += [
                Message(role="assistant", content=result.text),
                Message(
                    role="user",
                    content=f"Your reply was invalid ({last_error}). Reply with ONLY JSON.",
                ),
            ]
        return None, self.max_repairs + 1, f"malformed judgment: {last_error}"

    def _record(
        self,
        subject: str,
        mode: str,
        prompt: str,
        output: str,
        parsed: Any,
        input_tokens: int,
        output_tokens: int,
        cost: float | None,
        error: str | None,
        latency_ms: float | None = None,
    ) -> None:
        if self.store is None:
            return
        self.store.add_judge_call(
            run_id=self.run_id,
            subject=subject,
            judge_provider=self.provider_name,
            judge_model=self.model,
            mode=mode,
            prompt=prompt,
            output=output,
            parsed=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            error=error,
        )

    async def pointwise(
        self, question: str, candidate: str, reference: str | None, *, subject: str = ""
    ) -> JudgeVerdict:
        """Score one candidate answer; aggregates ``samples`` independent judgments."""
        system, user = self.rubric.pointwise_prompt(question, candidate, reference)
        mock = {
            "mode": "pointwise",
            "candidate": candidate,
            "reference": reference,
            "scale_min": self.rubric.scale_min,
            "scale_max": self.rubric.scale_max,
        }
        scores: list[int] = []
        reasoning = ""
        attempts = 0
        error: str | None = None
        for index in range(self.samples):
            parsed, used, err = await self._call(
                system,
                user,
                self._point_schema,
                mock,
                subject=subject,
                mode="pointwise",
                sample_index=index,
            )
            attempts += used
            if parsed is None:
                error = err
                continue
            scores.append(int(parsed["score"]))
            reasoning = reasoning or str(parsed.get("reasoning", ""))
        if not scores:
            return JudgeVerdict(
                score=None, normalized=None, passed=None, attempts=attempts, error=error
            )
        mean = sum(scores) / len(scores)
        modal = Counter(scores).most_common(1)[0][1] / len(scores)
        return JudgeVerdict(
            score=mean,
            normalized=self.rubric.normalise(mean),
            passed=mean >= self.rubric.pass_threshold,
            reasoning=reasoning,
            sample_scores=scores,
            self_consistency=modal if len(scores) > 1 else None,
            attempts=attempts,
            error=error,
        )

    async def _pair_once(
        self, question: str, a: str, b: str, reference: str | None, subject: str
    ) -> tuple[str | None, str | None]:
        system, user = self.rubric.pairwise_prompt(question, a, b, reference)
        mock = {"mode": "pairwise", "candidate_a": a, "candidate_b": b, "reference": reference}
        parsed, _, err = await self._call(
            system, user, self._pair_schema, mock, subject=subject, mode="pairwise", sample_index=0
        )
        return (str(parsed["winner"]) if parsed else None), err

    async def pairwise(
        self,
        question: str,
        answer_1: str,
        answer_2: str,
        reference: str | None,
        *,
        subject: str = "",
    ) -> PairwiseVerdict:
        """Compare two answers; with ``swap_order`` both orders are judged.

        The winner is reported in terms of the *inputs*: ``"1"``, ``"2"`` or ``"tie"``.
        If the two orders disagree the result is a tie and ``position_consistent`` is False.
        """
        first, err = await self._pair_once(question, answer_1, answer_2, reference, subject)
        first_norm = {"A": "1", "B": "2", "tie": "tie"}.get(first or "")
        if not self.swap_order:
            return PairwiseVerdict(
                winner=first_norm,
                first_order=first,
                swapped_order=None,
                position_consistent=None,
                error=err,
            )
        second, err2 = await self._pair_once(question, answer_2, answer_1, reference, subject)
        second_norm = {"A": "2", "B": "1", "tie": "tie"}.get(second or "")
        if first_norm is None or second_norm is None:
            return PairwiseVerdict(
                winner=None,
                first_order=first,
                swapped_order=second,
                position_consistent=None,
                error=err or err2,
            )
        consistent = first_norm == second_norm
        return PairwiseVerdict(
            winner=first_norm if consistent else "tie",
            first_order=first,
            swapped_order=second,
            position_consistent=consistent,
        )
