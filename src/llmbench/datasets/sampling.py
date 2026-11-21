"""Reproducible stratified sampling and splits."""

from __future__ import annotations

import random
from collections import defaultdict

from llmbench.datasets.schema import Sample


def _group_by_tag(samples: list[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.primary_tag].append(sample)
    return dict(sorted(groups.items()))


def stratified_sample(
    samples: list[Sample], n: int, *, seed: int = 0, stratify: bool = True
) -> list[Sample]:
    """Draw ``n`` samples reproducibly, allocating proportionally across primary tags.

    Allocation uses the largest-remainder method so every tag with enough mass
    is represented; the output preserves the dataset's original order.
    """
    if n >= len(samples):
        return list(samples)
    # Seeded, non-cryptographic RNG: reproducibility is the point.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    if not stratify:
        chosen = {id(s) for s in rng.sample(samples, n)}
        return [s for s in samples if id(s) in chosen]
    groups = _group_by_tag(samples)
    total = len(samples)
    quotas = {tag: n * len(items) / total for tag, items in groups.items()}
    alloc = {tag: int(q) for tag, q in quotas.items()}
    remainder = n - sum(alloc.values())
    for tag in sorted(quotas, key=lambda t: (alloc[t] - quotas[t], t))[:remainder]:
        alloc[tag] += 1
    picked: set[str] = set()
    for tag, items in groups.items():
        k = min(alloc[tag], len(items))
        picked.update(s.id for s in rng.sample(items, k))
    return [s for s in samples if s.id in picked]


def train_dev_split(
    samples: list[Sample], dev_fraction: float = 0.3, *, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Split into (train, dev) stratified by primary tag; used for judge calibration."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    # Seeded, non-cryptographic RNG: reproducibility is the point.
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    dev_ids: set[str] = set()
    for items in _group_by_tag(samples).values():
        shuffled = list(items)
        rng.shuffle(shuffled)
        k = round(len(shuffled) * dev_fraction)
        dev_ids.update(s.id for s in shuffled[:k])
    train = [s for s in samples if s.id not in dev_ids]
    dev = [s for s in samples if s.id in dev_ids]
    return train, dev
