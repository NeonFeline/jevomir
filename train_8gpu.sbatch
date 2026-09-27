#!/usr/bin/env bash
# Slurm wrapper for run_train_8gpu.sh: one whole hgx node (8x A100-80GB), everything on /raid.
#
#   sbatch train_8gpu.sbatch qlora --out runs/qlora-001 --no-4bit
#   sbatch train_8gpu.sbatch rft   --out runs/rft-001  --init-adapter runs/qlora-001/adapter
#   sbatch train_8gpu.sbatch grpo  --out runs/grpo-001 --init-adapter runs/rft-001/adapter --beta-kl 0.05
#
# Slurm log: /raid/$USER/jevomir/logs/slurm-<jobid>.out (the trainer also tees its own log there).
#SBATCH --job-name=jevomir-train
#SBATCH --partition=hgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=800G
#SBATCH --time=1-00:00:00
#SBATCH --output=/raid/%u/jevomir/logs/slurm-%j.out
set -euo pipefail

# sbatch copies the script to a spool dir; run from the submit directory (the repo).
cd "${SLURM_SUBMIT_DIR:?submit from the jevomir repo}"
echo "job $SLURM_JOB_ID on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-?}"
# torchrun ranks share this one Slurm task; a free port per job avoids clashes on shared nodes.
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
exec bash run_train_8gpu.sh "$@"
