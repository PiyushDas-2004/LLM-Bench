"""Inter-rater agreement: confusion matrices and Cohen's kappa (unweighted and weighted)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(
    truth: Sequence[Hashable], pred: Sequence[Hashable], labels: Sequence[Hashable]
) -> NDArray[np.int64]:
    """Counts with rows = truth label, columns = predicted label."""
    if len(truth) != len(pred):
        raise ValueError("truth and pred must have equal length")
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(truth, pred, strict=True):
        matrix[index[t], index[p]] += 1
    return matrix


def cohens_kappa(
    truth: Sequence[Hashable],
    pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
    weights: str | None = None,
) -> float:
    """Cohen's kappa = (p_o - p_e) / (1 - p_e).

    Args:
        truth: Reference ratings (e.g. human labels).
        pred: Ratings to compare (e.g. judge labels).
        labels: Ordered label set (required order for weighted kappa).
        weights: ``None`` (nominal), ``"linear"`` or ``"quadratic"`` (ordinal).

    Returns:
        Kappa in [-1, 1]; 1.0 when both raters use a single identical label.
    """
    labels = list(labels) if labels is not None else sorted(set(truth) | set(pred), key=str)
    observed = confusion_matrix(truth, pred, labels).astype(float)
    n = observed.sum()
    if n == 0:
        raise ValueError("no ratings")
    k = len(labels)
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / max(k - 1, 1)
    elif weights == "quadratic":
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    denom = float((w * expected).sum())
    if denom == 0:
        return 1.0
    return float(1.0 - (w * observed).sum() / denom)
