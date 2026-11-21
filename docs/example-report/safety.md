## LLMBench: safety-injection

Run `safety-injection-20261121T104514-47c1cac1` | task `safety` | dataset `safety_injection` (`bb9df6d3f6b4`) | 240/240 trials | status `completed`

> Mock provider: numbers are illustrative, not measurements of a real model.

| configuration | pass | injection_resistance | over_refusal | p50 latency | cost / 1k req |
|---|---:|---:|---:|---:|---:|
| `hardened/v1` (Pareto) | 0.967 [0.925, 1.000] | 0.974 [0.923, 1.000] | 0.048 [0.000, 0.143] | 398 ms | $0.0797 |
| `naive/v1` (Pareto) | 0.650 [0.533, 0.758] | 0.526 [0.372, 0.667] | 0.119 [0.024, 0.262] | 417 ms | $0.0147 |

Values: estimate [95% cluster-bootstrap CI].
