#!/usr/bin/env bash
# Robo2VLM-1 pipeline: review/clean -> one-pass extraction -> calibration probe.
# Usage: bash run_robo2vlm.sh   (override with DATA= OUT= FEAT= PROBE= BATCH= PY=)
set -euo pipefail

DATA=${DATA:-$HOME/agentic/robo2vlm/data}
OUT=${OUT:-runs/agentic}
FEAT=${FEAT:-runs/robo2vlm-features}
PROBE=${PROBE:-runs/robo2vlm-probe}
BATCH=${BATCH:-16}
PY=${PY:-.venv/bin/python}

mapfile -t PARQUETS < <(ls "$DATA"/*.parquet | sort)
echo "parquets: ${#PARQUETS[@]}"
"$PY" robo2vlm_tasks.py --parquet "${PARQUETS[@]}" --out "$OUT"

mkdir -p "$FEAT"
for p in "${PARQUETS[@]}"; do
    stem=$(basename "$p" .parquet)
    idx="$OUT/robo2vlm_${stem}.index.jsonl"
    if [ ! -f "$idx" ]; then
        echo "skip $stem (no index)"
        continue
    fi
    echo "=== extract $stem"
    "$PY" extract_robo2vlm.py --index "$idx" --parquet "$p" --out "$FEAT" --batch "$BATCH"
done

echo "=== probe"
rm -rf "$PROBE"
# Robo2VLM is a single source, so there is no subset to hold out (pass an empty list).
"$PY" train_probe_cauldron.py --features "$FEAT" --out "$PROBE" --holdout-subsets --epochs 300
echo "done: $PROBE/metrics.json"
