#!/bin/bash
#SBATCH --account=MST114208
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# Slurm payload -- not run directly. Submit via train_slurm.sh, which injects the
# job name, log paths and the config snapshot this trains from.
#
# Adjust before first use: --account / --partition / --time.

set -euo pipefail

CONFIG_PATH=${1:?usage: submit via train_slurm.sh <config_path> <run_name>}
RUN_NAME=${2:?missing <run_name>: run directory under weights/}
case "${RUN_NAME}" in -*) echo "run_name must not start with '-'" >&2; exit 1;; esac

cd "${SLURM_SUBMIT_DIR}"

# uv-managed virtualenv at the repo root (uv venv .venv && uv sync).
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK unset: launch via sbatch}
export MLFLOW_ALLOW_FILE_STORE=true
export MLFLOW_DISABLE_AGENT_HINT=1

python scripts/train.py \
  --config "${CONFIG_PATH}" \
  --save-dir "weights/${RUN_NAME}" \
  --run-name "${RUN_NAME}"

# Resume is automatic from weights/<run_name>/latest.pt (the MLflow run id rides in
# the checkpoint, so the same run continues). For a fresh start, use a new run_name.
