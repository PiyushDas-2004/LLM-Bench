"""Statistics: bootstrap CIs, paired tests, effect sizes, corrections, power, agreement.

Implemented directly on numpy/scipy so every formula is visible and tested;
see ``docs/METHODOLOGY.md``.
"""

from llmbench.stats.agreement import cohens_kappa, confusion_matrix
from llmbench.stats.bootstrap import (
    ConfidenceInterval,
    bca_ci,
    cluster_bootstrap_ci,
    cluster_bootstrap_replicates,
    percentile_ci,
)
from llmbench.stats.compare import MetricComparison, Verdict, apply_correction, compare_metric
from llmbench.stats.correction import adjust, benjamini_hochberg, holm
from llmbench.stats.effect import cliffs_delta, cohens_d, cohens_dz, relative_delta
from llmbench.stats.paired import (
    TestResult,
    mcnemar_test,
    paired_bootstrap,
    paired_permutation_test,
    wilcoxon_test,
)
from llmbench.stats.power import (
    PowerReport,
    minimum_detectable_effect,
    power_report,
    required_samples,
)

__all__ = [
    "ConfidenceInterval",
    "MetricComparison",
    "PowerReport",
    "TestResult",
    "Verdict",
    "adjust",
    "apply_correction",
    "bca_ci",
    "benjamini_hochberg",
    "cliffs_delta",
    "cluster_bootstrap_ci",
    "cluster_bootstrap_replicates",
    "cohens_d",
    "cohens_dz",
    "cohens_kappa",
    "compare_metric",
    "confusion_matrix",
    "holm",
    "mcnemar_test",
    "minimum_detectable_effect",
    "paired_bootstrap",
    "paired_permutation_test",
    "percentile_ci",
    "power_report",
    "relative_delta",
    "required_samples",
    "wilcoxon_test",
]
