#!/usr/bin/env bash
# Build the static documentation site with a freshly generated live demo.
# Everything runs offline against the mock provider: no API keys, no cost.
# Usage: scripts/build_site.sh [SITE_DIR]   (default: site)
set -euo pipefail

SITE_DIR="${1:-site}"
DEMO=docs/demo
WORK="$(mktemp -d)"
export LLMBENCH_DB_URL="sqlite:///${WORK}/demo.db"
export COLUMNS=160 NO_MKDOCS_2_WARNING=1
LLMBENCH="${LLMBENCH:-uv run llmbench}"

$LLMBENCH run examples/quickstart.yaml --yes > /dev/null
RUN="$($LLMBENCH list --name quickstart --limit 1 | grep -o 'quickstart-[0-9T]*-[0-9a-f]*' | head -1)"
$LLMBENCH judge calibrate datasets/judge_calibration.jsonl --config examples/quickstart.yaml \
  --attach "$RUN" --output "${WORK}/calibration.json" > /dev/null
$LLMBENCH report "$RUN" --output "${DEMO}/report.html"
$LLMBENCH report "$RUN" --format md --output "${DEMO}/quickstart.md"
$LLMBENCH compare "$RUN" "$RUN" --pair mock-fast/v1=mock-fast/v2 --pair mock-fast/v2=mock-strong/v2 \
  --metric pass --metric f1 --metric llm_judge_correctness --metric exact_agreement \
  --metric pass_hat_k --metric latency_ms --metric cost_usd --format md --output "${DEMO}/comparison.md"

$LLMBENCH import baselines/gate-demo.json --as-baseline main > /dev/null
$LLMBENCH run examples/gate/regression.yaml --yes > /dev/null
$LLMBENCH check --baseline main --gates examples/gates.yaml --output "${DEMO}/gate.md" > /dev/null || true

$LLMBENCH run examples/safety.yaml --yes > /dev/null
$LLMBENCH report latest --format md --output "${DEMO}/safety.md"
$LLMBENCH rag sweep examples/rag_sweep.yaml --top 10 --output "${DEMO}/rag_sweep.md" > /dev/null

python3 - "${WORK}/calibration.json" "${DEMO}/calibration.md" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
rows = [("Labelled examples scored", f"{r['n_scored']} / {r['n']}"), ("Accuracy", fmt(r["accuracy"])),
        ("Cohen's kappa (pass/fail)", fmt(r["cohen_kappa"])),
        ("Weighted kappa (1-5, quadratic)", fmt(r["weighted_kappa_quadratic"])),
        ("Spearman (scores)", fmt(r["spearman_scores"]))]
c = r["confusion"]
out = [f"## Judge calibration: `{r['judge']}` (rubric `{r['rubric']}`)", "",
       "> Mock judge (token-overlap scorer): illustrative only. Calibration exists to expose "
       "exactly this kind of weak judge before you trust it.", "",
       "| statistic | value |", "|---|---:|", *[f"| {k} | {v} |" for k, v in rows], "",
       "| human \\ judge | pass | fail |", "|---|---:|---:|",
       *[f"| {h} | {c[h]['pass']} | {c[h]['fail']} |" for h in ("pass", "fail")]]
open(sys.argv[2], "w").write("\n".join(out) + "\n")
PY

${MKDOCS:-uvx --from mkdocs --with mkdocs-material mkdocs} build --site-dir "$SITE_DIR"
rm -rf "$WORK"
echo "site built in ${SITE_DIR}"
