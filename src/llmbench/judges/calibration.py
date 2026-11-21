"""Calibrate a judge against human labels: accuracy, Cohen's kappa, confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from scipy import stats as sps

from llmbench.errors import DatasetError
from llmbench.judges.judge import LLMJudge
from llmbench.stats.agreement import cohens_kappa, confusion_matrix


class LabeledExample(BaseModel):
    """A human-labelled (question, reference, candidate) triple."""

    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    reference: str | None = None
    candidate: str
    label: Literal["pass", "fail"]
    score: int | None = None


class CalibrationItem(BaseModel):
    """Judge output for one labelled example."""

    id: str
    human_label: str
    judge_label: str | None
    human_score: int | None
    judge_score: float | None
    self_consistency: float | None


class CalibrationReport(BaseModel):
    """Agreement between the judge and human labels."""

    judge: str
    rubric: str
    n: int
    n_scored: int
    accuracy: float | None
    cohen_kappa: float | None
    weighted_kappa_quadratic: float | None
    spearman_scores: float | None
    confusion: dict[str, dict[str, int]]
    mean_self_consistency: float | None
    items: list[CalibrationItem] = Field(default_factory=list)


def load_labeled(path: str | Path) -> list[LabeledExample]:
    """Load a JSONL file of :class:`LabeledExample` records."""
    examples = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(LabeledExample.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DatasetError(f"{path}:{lineno}: {exc}") from exc
    if not examples:
        raise DatasetError(f"{path}: no labelled examples")
    return examples


async def calibrate(judge: LLMJudge, examples: list[LabeledExample]) -> CalibrationReport:
    """Run the judge on every example and compare with the human labels."""
    items: list[CalibrationItem] = []
    for ex in examples:
        verdict = await judge.pointwise(ex.question, ex.candidate, ex.reference, subject=ex.id)
        label = None if verdict.passed is None else ("pass" if verdict.passed else "fail")
        items.append(
            CalibrationItem(
                id=ex.id,
                human_label=ex.label,
                judge_label=label,
                human_score=ex.score,
                judge_score=verdict.score,
                self_consistency=verdict.self_consistency,
            )
        )
    scored = [i for i in items if i.judge_label is not None]
    truth = [i.human_label for i in scored]
    pred = [str(i.judge_label) for i in scored]
    labels = ["pass", "fail"]
    matrix = confusion_matrix(truth, pred, labels) if scored else None
    confusion = {
        t: {p: int(matrix[a, b]) if matrix is not None else 0 for b, p in enumerate(labels)}
        for a, t in enumerate(labels)
    }
    pairs = [
        (i.human_score, round(i.judge_score))
        for i in scored
        if i.human_score is not None and i.judge_score is not None
    ]
    rubric = judge.rubric
    scale = list(range(rubric.scale_min, rubric.scale_max + 1))
    weighted = (
        cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs], scale, weights="quadratic")
        if pairs
        else None
    )
    spearman = None
    if len(pairs) >= 3 and len({p[0] for p in pairs}) > 1 and len({p[1] for p in pairs}) > 1:
        spearman = float(sps.spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic)
    consistency = [i.self_consistency for i in scored if i.self_consistency is not None]
    return CalibrationReport(
        judge=f"{judge.provider_name}/{judge.model}",
        rubric=rubric.name,
        n=len(items),
        n_scored=len(scored),
        accuracy=sum(t == p for t, p in zip(truth, pred, strict=True)) / len(scored)
        if scored
        else None,
        cohen_kappa=cohens_kappa(truth, pred, labels) if scored else None,
        weighted_kappa_quadratic=weighted,
        spearman_scores=spearman,
        confusion=confusion,
        mean_self_consistency=sum(consistency) / len(consistency) if consistency else None,
        items=items,
    )
