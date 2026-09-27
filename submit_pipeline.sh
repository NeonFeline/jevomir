#!/usr/bin/env bash
# Overnight SFT -> RFT -> GRPO(+verbalized confidence) -> merge/eval pipeline on one hgx node.
#
#   bash submit_pipeline.sh [TAG]        # TAG defaults to pipe-YYYYmmdd-HHMM
#
# Snapshots this repo to /raid/$USER/jevomir/code/TAG (jobs run from there, so later edits in
# the repo cannot change a queued stage), then submits four Slurm jobs chained with afterok.
# Each training stage has a wall-clock budget (--time-budget-min): when it runs out the stage
# stops, evaluates and saves normally, so the whole chain fits in about 4 hours.
# Outputs: /raid/$USER/jevomir/runs/TAG/{sft,rft,grpo,final}; final/REPORT.md is the summary.
#
# Each stage trains on its own items: SFT on the first TRAIN items per subset, RFT on the next
# TRAIN, GRPO on the next TRAIN after that (--skip-train-subsets; images never shared).
# SFT_FROM=runs/OLD/sft reuses a finished SFT stage (symlinked) instead of training one.
set -euo pipefail

TAG=${1:-pipe-$(date +%Y%m%d-%H%M)}
NODE=${NODE:-hgx2}
JEV_ROOT=${JEV_ROOT:-/raid/$USER/jevomir}
CODE="$JEV_ROOT/code/$TAG"
RUN="runs/$TAG"

TRAIN="tallyqa:4000,nlvr2:4000,iconqa:3000,clevr:2500,vqav2:2500"
SKIP2="tallyqa:8000,nlvr2:8000,iconqa:6000,clevr:5000,vqav2:5000"  # 2 x TRAIN: SFT's and RFT's items
EVAL="tallyqa:300,nlvr2:300,iconqa:300,clevr:300,vqav2:300"
COMMON=(--no-4bit --train-subsets "$TRAIN" --eval-subsets "$EVAL" --eval-batch 32 --eval-workers 6
        --workers 10 --shuffle-options --log-every 10 --seed 0)

src=$(cd "$(dirname "$0")" && pwd)
if [ -e "$CODE" ]; then echo "$CODE exists; pick another TAG" >&2; exit 1; fi
mkdir -p "$CODE" "$JEV_ROOT/logs"
rsync -a --exclude .git --exclude .venv --exclude runs --exclude artifacts --exclude web --exclude tetris \
    "$src/" "$CODE/"
git -C "$src" rev-parse HEAD > "$CODE/GIT_HEAD"
git -C "$src" diff HEAD > "$CODE/GIT_DIFF.patch" || true
cd "$CODE"

sub() {  # sub NAME TIME DEPENDENCY-JOBID args...
    local name=$1 time=$2 dep=$3; shift 3
    sbatch --parsable --job-name="jev-$name" --nodelist="$NODE" --exclusive --time="$time" \
        ${dep:+--dependency=afterok:$dep --kill-on-invalid-dep=yes} train_8gpu.sbatch "$@"
}

if [ -n "${SFT_FROM:-}" ]; then
    mkdir -p "$JEV_ROOT/$RUN"
    ln -sfn "$JEV_ROOT/$SFT_FROM" "$JEV_ROOT/$RUN/sft"
    j1=""
else
    j1=$(sub sft 01:40:00 "" qlora --out "$RUN/sft" "${COMMON[@]}" \
        --epochs 2 --batch 8 --grad-accum 1 --lr 2e-4 --warmup 20 --save-every 100 --time-budget-min 40)
fi
j2=$(sub rft 01:40:00 "$j1" rft --out "$RUN/rft" --init-adapter "$RUN/sft/adapter" "${COMMON[@]}" \
    --skip-train-subsets "$TRAIN" \
    --rounds 2 --k 8 --sft-epochs 1 --batch 8 --grad-accum 1 --lr 1e-4 --warmup 10 \
    --rollout-batch 32 --time-budget-min 15)
# Confidence: proper score + distillation of the letter probability (per-item signal); answer
# policy anchored by KL to the RFT policy, at a lower lr than the overnight 1e-4 run.
j3=$(sub grpo 02:15:00 "$j2" grpo --out "$RUN/grpo" --init-adapter "$RUN/rft/adapter" "${COMMON[@]}" \
    --skip-train-subsets "$SKIP2" \
    --confidence --conf-temperature 4 --lambda-consistency 0.5 --lambda-conf 0.5 \
    --lambda-distill 1.0 --beta-kl 0.1 --kl-ref init \
    --epochs 1 --batch 4 --grad-accum 1 --lr 3e-5 --warmup 10 --save-every 50 --time-budget-min 75)
j4=$(sub final 00:50:00 "$j3" finalize --runs "$RUN" --adapter "$RUN/grpo/adapter" --out "$RUN/final" \
    --eval-subsets "$EVAL" --eval-batch 32 --confidence)

echo "$TAG: sft=$j1 rft=$j2 grpo=$j3 final=$j4" | tee -a "$JEV_ROOT/logs/pipelines.txt"
echo "code snapshot: $CODE"
echo "outputs:       $JEV_ROOT/$RUN  (report: $JEV_ROOT/$RUN/final/REPORT.md)"
