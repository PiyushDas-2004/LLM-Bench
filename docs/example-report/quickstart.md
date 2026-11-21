## LLMBench: quickstart

Run `quickstart-20261121T104455-19da8e05` | task `qa` | dataset `qa_basic` (`31c0e369dbf5`) | 800/800 trials | status `completed`

> Mock provider: numbers are illustrative, not measurements of a real model.

| configuration | pass | exact_match | f1 | llm_judge_correctness | p50 latency | cost / 1k req |
|---|---:|---:|---:|---:|---:|---:|
| `mock-fast/v1` (Pareto) | 0.705 [0.590, 0.805] | 0.700 [0.585, 0.795] | 0.706 [0.590, 0.803] | 0.634 [0.524, 0.739] | 187 ms | $0.0043 |
| `mock-fast/v2` (Pareto) | 0.765 [0.670, 0.855] | 0.765 [0.670, 0.855] | 0.768 [0.672, 0.860] | 0.666 [0.569, 0.765] | 179 ms | $0.0055 |
| `mock-strong/v1` (Pareto) | 0.825 [0.735, 0.905] | 0.825 [0.735, 0.905] | 0.825 [0.735, 0.905] | 0.713 [0.617, 0.805] | 694 ms | $0.0215 |
| `mock-strong/v2` (Pareto) | 0.855 [0.770, 0.925] | 0.855 [0.770, 0.925] | 0.855 [0.770, 0.925] | 0.740 [0.650, 0.819] | 628 ms | $0.0278 |

Values: estimate [95% cluster-bootstrap CI].
