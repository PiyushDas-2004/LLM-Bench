from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llmbench.stats import (
    Verdict,
    adjust,
    apply_correction,
    bca_ci,
    benjamini_hochberg,
    cliffs_delta,
    cluster_bootstrap_ci,
    cohens_d,
    cohens_dz,
    compare_metric,
    holm,
    mcnemar_test,
    minimum_detectable_effect,
    paired_bootstrap,
    paired_permutation_test,
    percentile_ci,
    power_report,
    relative_delta,
    required_samples,
    wilcoxon_test,
)
from llmbench.stats.effect import magnitude_label


def test_percentile_and_bca_contain_mean() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(5, 1, 200)
    for fn in (percentile_ci, bca_ci):
        ci = fn(x, n_resamples=1000)
        assert ci.low < 5 < ci.high
        assert ci.contains(ci.estimate)
        assert ci.width > 0
    median = bca_ci(x, statistic=lambda a: float(np.median(a)), n_resamples=500)
    assert median.low < median.estimate < median.high
    pct_median = percentile_ci(x, statistic=lambda a: float(np.median(a)), n_resamples=200)
    assert pct_median.method == "percentile"
    degenerate = bca_ci([1.0, 1.0, 1.0])
    assert degenerate.low == degenerate.high == 1.0
    with pytest.raises(ValueError, match="non-empty"):
        percentile_ci([])


@pytest.mark.slow
def test_bootstrap_coverage_is_near_nominal() -> None:
    """Over many synthetic datasets the 95% CI should contain the true mean ~95% of the time."""
    rng = np.random.default_rng(42)
    hits = 0
    reps = 300
    for i in range(reps):
        x = rng.exponential(1.0, 60)  # skewed, true mean 1.0
        hits += bca_ci(x, n_resamples=500, seed=i).contains(1.0)
    coverage = hits / reps
    assert 0.89 <= coverage <= 0.99


def test_cluster_bootstrap_is_wider_for_correlated_trials() -> None:
    rng = np.random.default_rng(0)
    # 30 samples with very different difficulty, 5 nearly identical trials each
    clusters = [list(np.clip(rng.normal(p, 0.01, 5), 0, 1)) for p in rng.uniform(0, 1, 30)]
    flat = [v for c in clusters for v in c]
    cluster_ci = cluster_bootstrap_ci(clusters, n_resamples=1000)
    naive = percentile_ci(flat, n_resamples=1000)
    assert cluster_ci.width > 1.8 * naive.width
    assert cluster_ci.n_samples == 30
    assert cluster_ci.n_trials == 150
    mapping = cluster_bootstrap_ci({"a": [1.0, 0.0], "b": [1.0]}, n_resamples=100)
    assert mapping.estimate == pytest.approx(0.75)
    with pytest.raises(ValueError, match="at least one"):
        cluster_bootstrap_ci([[1.0], []])


def test_permutation_test_exact_and_monte_carlo() -> None:
    a = [0.0] * 10
    b = [1.0] * 10
    exact = paired_permutation_test(a, b)
    assert exact.p_value == pytest.approx(2 / 2**10)
    assert paired_permutation_test(a, a).p_value == 1.0
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 50)
    big = paired_permutation_test(x, x + 1.0, n_resamples=2000)
    assert big.p_value < 0.01
    with pytest.raises(ValueError, match="equal length"):
        paired_permutation_test([1.0], [1.0, 2.0])


@pytest.mark.slow
def test_permutation_p_values_are_uniform_under_null() -> None:
    rng = np.random.default_rng(7)
    ps = []
    for i in range(400):
        a = rng.normal(0, 1, 30)
        b = rng.normal(0, 1, 30)
        ps.append(paired_permutation_test(a, b, n_resamples=500, seed=i).p_value)
    ps_arr = np.array(ps)
    false_positive_rate = float(np.mean(ps_arr < 0.05))
    assert 0.02 <= false_positive_rate <= 0.09
    # roughly uniform: each decile holds 10% +- 5%
    hist, _ = np.histogram(ps_arr, bins=10, range=(0, 1))
    assert np.all(np.abs(hist / len(ps) - 0.1) < 0.05)


def test_paired_bootstrap() -> None:
    a = np.zeros(40)
    b = np.r_[np.ones(30), np.zeros(10)]
    ci, test = paired_bootstrap(a, b, n_resamples=1000)
    assert ci.low > 0
    assert test.p_value < 0.01
    _, null = paired_bootstrap(a, a, n_resamples=200)
    assert null.p_value == 1.0


def test_mcnemar() -> None:
    assert mcnemar_test([1, 1], [1, 1]).p_value == 1.0
    small = mcnemar_test([1, 1, 1, 1, 0], [0, 0, 0, 0, 0])
    assert small.test == "mcnemar-exact"
    assert small.p_value == pytest.approx(0.125)
    a = [1] * 40 + [0] * 10
    b = [0] * 40 + [0] * 10
    large = mcnemar_test(a, b)
    assert large.test == "mcnemar-chi2"
    assert large.p_value < 1e-6
    with pytest.raises(ValueError, match="equal length"):
        mcnemar_test([1], [])


def test_wilcoxon() -> None:
    assert wilcoxon_test([1.0, 2.0], [1.0, 2.0]).p_value == 1.0
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 30)
    assert wilcoxon_test(x, x + 2).p_value < 0.001


def test_effect_sizes() -> None:
    assert cohens_dz([0, 0, 0], [1, 2, 3]) == pytest.approx(2.0)
    assert cohens_dz([1], [2]) == 0.0
    assert cohens_dz([0, 0], [0, 0]) == 0.0
    assert cohens_dz([0, 0], [1, 1]) == np.inf
    assert cohens_d([0, 1, 2], [1, 2, 3]) == pytest.approx(1.0)
    assert cohens_d([1], [1, 2]) == 0.0
    assert cohens_d([1, 1], [1, 1]) == 0.0
    assert cliffs_delta([0, 0], [1, 1]) == 1.0
    assert cliffs_delta([1, 1], [0, 0]) == -1.0
    assert cliffs_delta([], [1]) == 0.0
    assert relative_delta(0.5, 0.6) == pytest.approx(0.2)
    assert relative_delta(0, 1) is None
    assert [magnitude_label(d) for d in (0.1, 0.3, 0.6, 1.2)] == [
        "negligible",
        "small",
        "medium",
        "large",
    ]


def test_corrections_known_values() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert adjust(p, "none") == p
    assert adjust(p, "bh") == benjamini_hochberg(p)
    assert holm([]) == []
    assert benjamini_hochberg([]) == []
    with pytest.raises(ValueError, match="unknown correction"):
        adjust(p, "magic")


@given(st.lists(st.floats(0, 1), min_size=1, max_size=20))
def test_corrections_are_monotone_and_bounded(p: list[float]) -> None:
    for adjusted in (holm(p), benjamini_hochberg(p)):
        assert all(0 <= q <= 1 for q in adjusted)
        assert all(q >= raw - 1e-12 for q, raw in zip(adjusted, p, strict=True))
    assert all(h >= b - 1e-12 for h, b in zip(holm(p), benjamini_hochberg(p), strict=True))


def test_power() -> None:
    mde = minimum_detectable_effect(0.5, 100)
    assert mde == pytest.approx(2.8016 * 0.05, rel=1e-3)
    assert minimum_detectable_effect(0.5, 0) == np.inf
    n = required_samples(0.5, mde)
    assert 99 <= n <= 101
    with pytest.raises(ValueError, match="positive"):
        required_samples(0.5, 0)
    small = power_report([0, 1, 0], [1, 1, 0])
    assert small.warning is not None
    assert "paired samples" in small.warning
    flat = power_report([1] * 40, [1] * 40, binary=True)
    assert flat.sd_diff == 0.5
    assert flat.warning is None


def _per_sample(values: list[list[float]]) -> dict[str, list[float]]:
    return {f"s{i}": v for i, v in enumerate(values)}


def test_compare_metric_regression_and_improvement() -> None:
    rng = np.random.default_rng(0)
    base = _per_sample([[float(rng.random() < 0.85) for _ in range(5)] for _ in range(60)])
    worse = _per_sample([[float(rng.random() < 0.5) for _ in range(5)] for _ in range(60)])
    cmp = compare_metric("pass", base, worse, kind="binary")
    assert cmp.verdict is Verdict.REGRESSED
    assert cmp.test == "paired-permutation"
    assert cmp.delta is not None
    assert cmp.delta < 0
    assert cmp.baseline is not None
    assert cmp.baseline.n_trials == 300
    assert cmp.effect_size_name == "cliffs_delta"
    assert compare_metric("pass", worse, base, kind="binary").verdict is Verdict.IMPROVED
    latency = compare_metric("latency", base, worse, higher_is_better=False)
    assert latency.verdict is Verdict.IMPROVED
    assert latency.effect_magnitude is not None


def test_compare_metric_test_selection_and_edge_cases() -> None:
    single_b = _per_sample([[1.0]] * 30 + [[0.0]] * 10)
    single_c = _per_sample([[1.0]] * 30 + [[0.0]] * 10)
    cmp = compare_metric("pass", single_b, single_c, kind="binary")
    assert cmp.test.startswith("mcnemar")
    assert cmp.verdict is Verdict.NO_CHANGE
    for test in ("wilcoxon", "bootstrap", "permutation"):
        assert compare_metric("x", single_b, single_c, test=test).verdict is Verdict.NO_CHANGE
    with pytest.raises(ValueError, match="unknown test"):
        compare_metric("x", single_b, single_c, test="t-test")
    tiny = compare_metric("x", {"a": [1.0]}, {"a": [0.0], "b": [1.0]})
    assert tiny.verdict is Verdict.INSUFFICIENT
    assert tiny.warning is not None


def test_apply_correction_can_flip_verdicts() -> None:
    rng = np.random.default_rng(5)
    base = _per_sample([[float(rng.random() < 0.6)] for _ in range(80)])
    cand = _per_sample([[float(rng.random() < 0.75)] for _ in range(80)])
    raw = compare_metric("m", base, cand, kind="binary", test="permutation")
    family = [raw] + [raw.model_copy(update={"metric": f"m{i}"}) for i in range(20)]
    corrected = apply_correction(family, "holm")
    assert all(c.p_adjusted is not None and c.p_adjusted >= (c.p_value or 0) for c in corrected)
    if raw.p_value is not None and raw.p_value * 21 >= 0.05:
        assert corrected[0].verdict is Verdict.NO_CHANGE


@settings(deadline=None, max_examples=25)
@given(st.lists(st.integers(0, 1), min_size=5, max_size=30))
def test_identical_configs_never_regress(outcomes: list[int]) -> None:
    per = {f"s{i}": [float(v)] for i, v in enumerate(outcomes)}
    cmp = compare_metric("pass", per, per, kind="binary", n_resamples=200)
    assert cmp.verdict in {Verdict.NO_CHANGE, Verdict.INSUFFICIENT}
    assert cmp.delta == 0
