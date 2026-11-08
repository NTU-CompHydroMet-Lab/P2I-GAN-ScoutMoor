#!/bin/bash
# Local (no-slurm) training with the same weights/<run_name>/ layout as
# train_slurm.sh: config snapshot in logs/, console tee'd to logs/.
#
#   Usage: scripts/train_jobs/train_local.sh <config_path> <run_name>
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG_PATH=${1:?usage: train_local.sh <config_path> <run_name>}
RUN_NAME=${2:?missing <run_name>: run directory under weights/}
[ $# -eq 2 ] || { echo "expected exactly 2 args, got $#" >&2; exit 1; }
case "${RUN_NAME}" in -*) echo "run_name must not start with '-'" >&2; exit 1;; esac

RUN_DIR="weights/${RUN_NAME}"
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "${RUN_DIR}/logs"
CONFIG_SNAP="${RUN_DIR}/logs/config-${STAMP}.json"
cp "${CONFIG_PATH}" "${CONFIG_SNAP}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Online by default; export WANDB_MODE=offline to log locally and push later with `wandb sync`.
export WANDB_MODE=${WANDB_MODE:-online}

uv run python scripts/train.py \
  --config "${CONFIG_SNAP}" \
  --save-dir "${RUN_DIR}" \
  --run-name "${RUN_NAME}" \
  2>&1 | tee -a "${RUN_DIR}/logs/${RUN_NAME}-local-${STAMP}.out"
