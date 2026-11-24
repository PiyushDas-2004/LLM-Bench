#!/usr/bin/env bash
# Build the static documentation site with a freshly generated live demo.
# Everything runs offline against the mock provider: no API keys, no cost.
# Usage: scripts/build_site.sh [SITE_DIR]   (default: site)
set -euo pipefail

SITE_DIR="${1:-site}"
DEMO=docs/demo
DATA=frontend/public/data
mkdir -p "$DEMO" "$DATA"
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
$LLMBENCH report "$RUN" --format json --output "${DATA}/quickstart.json"
COMPARE_ARGS=(--pair mock-fast/v1=mock-fast/v2 --pair mock-fast/v2=mock-strong/v2
  --metric pass --metric f1 --metric llm_judge_correctness --metric exact_agreement
  --metric pass_hat_k --metric latency_ms --metric cost_usd)
$LLMBENCH compare "$RUN" "$RUN" "${COMPARE_ARGS[@]}" --format md --output "${DEMO}/comparison.md"
$LLMBENCH compare "$RUN" "$RUN" "${COMPARE_ARGS[@]}" --format json --output "${DATA}/comparison.json"

$LLMBENCH import baselines/gate-demo.json --as-baseline main > /dev/null
$LLMBENCH run examples/gate/regression.yaml --yes > /dev/null
$LLMBENCH check --baseline main --gates examples/gates.yaml --output "${DEMO}/gate.md" \
  --json "${DATA}/gate.json" > /dev/null || true

$LLMBENCH run examples/safety.yaml --yes > /dev/null
$LLMBENCH report latest --format md --output "${DEMO}/safety.md"
$LLMBENCH report latest --format json --output "${DATA}/safety.json"
$LLMBENCH rag sweep examples/rag_sweep.yaml --top 10 --output "${DEMO}/rag_sweep.md" \
  --json "${WORK}/rag_sweep.json" > /dev/null

# Dashboard data: drop per-question detail from the sweep to keep the page light.
python3 - "${WORK}" "${DATA}" <<'PY'
import datetime, json, os, subprocess, sys
work, data = sys.argv[1], sys.argv[2]
sweep = json.load(open(f"{work}/rag_sweep.json"))
for result in sweep["results"]:
    result.pop("questions", None)
json.dump(sweep, open(f"{data}/rag_sweep.json", "w"))
cal = json.load(open(f"{work}/calibration.json"))
cal.pop("items", None)
json.dump(cal, open(f"{data}/calibration.json", "w"))
commit = os.environ.get("GITHUB_SHA")
if not commit:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
version = subprocess.run(["uv", "run", "python", "-c", "import llmbench; print(llmbench.__version__)"],
                         capture_output=True, text=True).stdout.strip() or "unknown"
json.dump({"generated_at": datetime.datetime.now(datetime.UTC).isoformat(), "commit": commit,
           "llmbench_version": version}, open(f"{data}/manifest.json", "w"))
PY

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

# Web dashboard (React + TypeScript), served from <site>/app/.
(cd frontend && npm ci --no-audit --no-fund && npm run build)
mkdir -p "${SITE_DIR}/app"
cp -R frontend/dist/. "${SITE_DIR}/app/"

rm -rf "$WORK"
echo "site built in ${SITE_DIR} (dashboard in ${SITE_DIR}/app)"
