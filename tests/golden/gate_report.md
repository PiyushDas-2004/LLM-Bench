## LLMBench eval gate: FAILED

Baseline `<run-id>` vs candidate `<run-id>`

| gate | configuration | result | reason |
|---|---|---|---|
| pass | `assistant/v1` | **FAIL** | regressed by 0.2417 (> max_drop 0.02); p_adj=0.0003 < 0.05 with paired-permutation |

## LLMBench comparison: `<run-id>` -> `<run-id>`

alpha = 0.05, correction = holm, dataset match = True

### `assistant/v1` -> `assistant/v1`

| metric | baseline | candidate | delta [95% CI] | test | p (adj.) | effect size | n | verdict |
|---|---:|---:|---:|---|---:|---:|---:|---|
| pass | 0.817 | 0.575 | -0.242 [-0.359, -0.125] | paired-permutation | <0.001 | cliffs_delta=-0.40 | 40 x 3 | **regressed** |


<details><summary>Per-sample regressions</summary>

| sample | baseline | candidate | baseline output | candidate output |
|---|---:|---:|---|---|
| qa-018 | 1.00 | 0.00 | Neon | approximately |
| qa-034 | 1.00 | 0.00 | 1912 | 1915 |
| qa-040 | 1.00 | 0.00 | went | approximately |
| qa-019 | 1.00 | 0.33 | Mitochondria | several |
| qa-022 | 1.00 | 0.33 | 13 | 14 |

</details>

| tag | n | baseline | candidate | delta |
|---|---:|---:|---:|---:|
| geography | 9 | 0.815 | 0.556 | -0.259 |
| history | 8 | 0.833 | 0.667 | -0.167 |
| language | 2 | 0.833 | 0.167 | -0.667 |
| math | 8 | 0.708 | 0.542 | -0.167 |
| science | 10 | 0.867 | 0.600 | -0.267 |
| units | 3 | 0.889 | 0.667 | -0.222 |

