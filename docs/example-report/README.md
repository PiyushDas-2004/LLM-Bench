# Example report (mock provider, illustrative)

Everything in this folder was generated on **2026-11-21** by running LLMBench in this
repository with **LLMBench's deterministic mock provider**. No API keys were available in
the environment that produced it, so **no real model was called**. The numbers show what
the reports look like and exercise every statistic end to end; they are not measurements
of any real model.

| File | Command that produced it |
|---|---|
| [`quickstart.html`](quickstart.html) | `llmbench run examples/quickstart.yaml` then `llmbench report <run> -o docs/example-report/quickstart.html` |
| [`quickstart.md`](quickstart.md) | `llmbench report <run> --format md` |
| [`comparison.md`](comparison.md) | `llmbench compare <run> <run> --pair mock-fast/v1=mock-fast/v2 --pair mock-fast/v2=mock-strong/v2 --metric ... --format md` |
| [`judge_calibration.json`](judge_calibration.json), [`.txt`](judge_calibration.txt) | `llmbench judge calibrate datasets/judge_calibration.jsonl --config examples/quickstart.yaml --attach <run>` |
| [`safety.md`](safety.md) | `llmbench run examples/safety.yaml` then `llmbench report latest --format md` |
| [`rag_sweep.md`](rag_sweep.md) | `llmbench rag sweep examples/rag_sweep.yaml --output docs/example-report/rag_sweep.md --top 8` |
| [`report-screenshot.png`](report-screenshot.png) | top of `quickstart.html`, light theme |

How to read the mock numbers: the mock's `accuracy` knob is a target on the logit scale
with per-sample difficulty, so observed pass rates differ from the nominal value (see
[METHODOLOGY.md](../METHODOLOGY.md)). The mock judge scores answers by token overlap with
the reference, which is why its calibration against the human labels is only moderate
(Cohen's kappa 0.333 on 40 labelled examples in this run): it penalises correct but
verbose answers. That is exactly the kind of weakness calibration is meant to expose.

To produce a real report, export `OPENAI_API_KEY` and/or `GEMINI_API_KEY`, set real
model names in `examples/experiment.yaml`, and run `llmbench run examples/experiment.yaml`.
