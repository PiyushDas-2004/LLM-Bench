## LLMBench comparison: `quickstart-20261121T104455-19da8e05` -> `quickstart-20261121T104455-19da8e05`

alpha = 0.05, correction = holm, dataset match = True

### `mock-fast/v1` -> `mock-fast/v2`

| metric | baseline | candidate | delta [95% CI] | test | p (adj.) | effect size | n | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|
| pass | 0.705 | 0.765 | +0.060 [-0.010, +0.135] | paired-permutation | 0.671 | cliffs_delta=0.10 | 40 x 5 | no significant change |
| f1 | 0.706 | 0.768 | +0.062 [-0.0075, +0.135] | paired-permutation | 0.671 | cohens_dz=0.27 | 40 x 5 | no significant change |
| llm_judge_correctness | 0.634 | 0.666 | +0.032 [-0.041, +0.105] | paired-permutation | 1.000 | cohens_dz=0.14 | 40 x 5 | no significant change |
| exact_agreement | 0.665 | 0.665 | +0.000 [-0.070, +0.075] | paired-permutation | 1.000 | cohens_dz=0.00 | 40 x 1 | no significant change |
| pass_hat_k | 0.483 | 0.543 | +0.060 [-0.068, +0.192] | paired-permutation | 1.000 | cohens_dz=0.15 | 40 x 1 | no significant change |
| latency_ms | 199.266 | 187.518 | -11.747 [-26.058, +2.741] | paired-permutation | 0.671 | cohens_dz=-0.25 | 40 x 5 | no significant change |
| cost_usd | 4.32e-06 | 5.54e-06 | +1.23e-06 [+1.19e-06, +1.26e-06] | paired-permutation | <0.001 | cohens_dz=11.69 | 40 x 5 | **regressed** |


<details><summary>Per-sample regressions</summary>

| sample | baseline | candidate | baseline output | candidate output |
|---|---:|---:|---|---|
| qa-027 | 0.80 | 0.40 | 178 | 178 |
| qa-036 | 1.00 | 0.60 | 100 | 100 |
| qa-013 | 0.80 | 0.60 | 6 | 4 |
| qa-020 | 0.80 | 0.60 | 56 | 54 |
| qa-007 | 0.60 | 0.40 | Amazon | several |

</details>

| tag | n | baseline | candidate | delta |
|---|---:|---:|---:|---:|
| geography | 9 | 0.733 | 0.800 | +0.067 |
| history | 8 | 0.800 | 0.750 | -0.050 |
| language | 2 | 0.600 | 0.600 | +0.000 |
| math | 8 | 0.500 | 0.700 | +0.200 |
| science | 10 | 0.820 | 0.920 | +0.100 |
| units | 3 | 0.600 | 0.467 | -0.133 |

### `mock-fast/v2` -> `mock-strong/v2`

| metric | baseline | candidate | delta [95% CI] | test | p (adj.) | effect size | n | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|
| pass | 0.765 | 0.855 | +0.090 [-0.015, +0.190] | paired-permutation | 0.497 | cliffs_delta=0.20 | 40 x 5 | no significant change |
| f1 | 0.768 | 0.855 | +0.088 [-0.015, +0.185] | paired-permutation | 0.497 | cohens_dz=0.27 | 40 x 5 | no significant change |
| llm_judge_correctness | 0.666 | 0.740 | +0.074 [-0.030, +0.173] | paired-permutation | 0.497 | cohens_dz=0.22 | 40 x 5 | no significant change |
| exact_agreement | 0.665 | 0.720 | +0.055 [-0.050, +0.150] | paired-permutation | 0.497 | cohens_dz=0.18 | 40 x 1 | no significant change |
| pass_hat_k | 0.543 | 0.705 | +0.162 [-0.030, +0.350] | paired-permutation | 0.497 | cohens_dz=0.27 | 40 x 1 | no significant change |
| latency_ms | 187.518 | 675.467 | +487.949 [+448.094, +530.702] | paired-permutation | <0.001 | cohens_dz=3.66 | 40 x 5 | **regressed** |
| cost_usd | 5.54e-06 | 2.78e-05 | +2.22e-05 [+2.16e-05, +2.29e-05] | paired-permutation | <0.001 | cohens_dz=10.69 | 40 x 5 | **regressed** |


<details><summary>Per-sample regressions</summary>

| sample | baseline | candidate | baseline output | candidate output |
|---|---:|---:|---|---|
| qa-002 | 1.00 | 0.40 | tokyo | I am not sure. |
| qa-035 | 1.00 | 0.40 | in 1957 | 1956 |
| qa-015 | 1.00 | 0.60 | Mars | perhaps |
| qa-017 | 1.00 | 0.60 | Na | different |
| qa-025 | 0.40 | 0.00 | 391 | 389 |

</details>

| tag | n | baseline | candidate | delta |
|---|---:|---:|---:|---:|
| geography | 9 | 0.800 | 0.889 | +0.089 |
| history | 8 | 0.750 | 0.900 | +0.150 |
| language | 2 | 0.600 | 0.700 | +0.100 |
| math | 8 | 0.700 | 0.750 | +0.050 |
| science | 10 | 0.920 | 0.860 | -0.060 |
| units | 3 | 0.467 | 1.000 | +0.533 |

