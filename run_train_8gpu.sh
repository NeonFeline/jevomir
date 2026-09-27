#!/usr/bin/env bash
# Data-parallel (Q)LoRA / RFT / GRPO on one 8xA100 node. Everything lives under JEV_ROOT on /raid.
#
#   bash run_train_8gpu.sh qlora --out runs/qlora-001 [train_qlora.py args ...]
#   bash run_train_8gpu.sh rft   --out runs/rft-001   --init-adapter runs/qlora-001/adapter
#   bash run_train_8gpu.sh grpo  --out runs/grpo-001  --init-adapter runs/rft-001/adapter --beta-kl 0.05
#   bash run_train_8gpu.sh qlora --out runs/smoke-q --smoke
#
# Relative --out / --init-adapter paths resolve under JEV_ROOT (default /raid/$USER/jevomir), e.g.
# runs/qlora-001 -> /raid/$USER/jevomir/runs/qlora-001. Data defaults to JEV_ROOT/cauldron.
# Global batch = --batch x --grad-accum x NGPU (defaults 4 x 4 x 8 = 128 for qlora): scale up
# --train-subsets or lower --grad-accum vs single-GPU runs; the scripts print the step count and
# refuse to start when the pool is smaller than one global batch. 4B in bf16 fits an 80GB A100:
# --no-4bit (plain LoRA) is noticeably faster than QLoRA's per-matmul dequantization.
#
# Env overrides: JEV_ROOT, NGPU (8), PY (JEV_ROOT/venv/bin/python), MASTER_PORT (29517).
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,16p' "$0"
    exit 1
fi
case "$1" in
    qlora|rft|grpo) SCRIPT="train_$1.py" ;;
    finalize) SCRIPT="finalize_pipeline.py" ;;
    extract) SCRIPT="extract_finetuned.py" ;;
    *) echo "unknown trainer '$1' (qlora|rft|grpo|finalize)" >&2; exit 1 ;;
esac
shift

cd "$(dirname "$0")"
JEV_ROOT=${JEV_ROOT:-/raid/$USER/jevomir}
NGPU=${NGPU:-8}
PY=${PY:-$JEV_ROOT/venv/bin/python}
MASTER_PORT=${MASTER_PORT:-29517}
mkdir -p "$JEV_ROOT/runs" "$JEV_ROOT/logs" "$JEV_ROOT/triton" "$JEV_ROOT/hf/hub"

# Resolve relative path arguments under JEV_ROOT so no output lands in $HOME.
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --out|--init-adapter|--data|--runs|--adapter|--save-model)
            v=$2; [[ "$v" = /* ]] || v="$JEV_ROOT/$v"
            args+=("$1" "$v"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

export JEV_DATA=${JEV_DATA:-$JEV_ROOT/cauldron}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$JEV_ROOT/hf/hub}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}        # 128 cores / 8 ranks; DataLoader workers do the CPU work
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1            # NCCL failures raise instead of hanging
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TRITON_CACHE_DIR_BASE="$JEV_ROOT/triton"     # per-rank: 8 ranks autotuning one dir races

visible=$("$PY" -c "import torch; print(torch.cuda.device_count())")
if [ "$visible" -lt "$NGPU" ]; then
    echo "need $NGPU GPUs, see $visible (set NGPU or CUDA_VISIBLE_DEVICES)" >&2
    exit 1
fi

log="$JEV_ROOT/logs/$(basename "$SCRIPT" .py)-$(date +%Y%m%d-%H%M%S).log"
echo "launching $SCRIPT on $NGPU GPUs, root $JEV_ROOT, log -> $log"
"$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" --master_port "$MASTER_PORT" \
    --no-python bash -c 'TRITON_CACHE_DIR="$TRITON_CACHE_DIR_BASE/rank$LOCAL_RANK" exec "$0" "$@"' \
    "$PY" "$SCRIPT" "${args[@]}" 2>&1 | tee "$log"
exit "${PIPESTATUS[0]}"
