#!/bin/bash
# Submit a training job on slurm with per-run log paths and a config snapshot.
# #SBATCH directives cannot reference variables, so the run-specific job name and
# log paths are injected here at submit time.
#
#   Usage: scripts/train_jobs/train_slurm.sh <config_path> <run_name>
#     config_path  JSON config file
#     run_name     run directory under weights/ and MLflow run name
#
# The job trains from the CONFIG SNAPSHOT, not the live config: editing the JSON
# while the job sits in the queue cannot change what it runs.
#
# Everything for one run lands under weights/<run_name>/:
#   latest.pt  best.pt  artifacts/  logs/{<run_name>-<jobid>.out,.err}
#   logs/config-<stamp>.json  logs/sbatch_script-<stamp>.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG_PATH=${1:?usage: train_slurm.sh <config_path> <run_name>}
RUN_NAME=${2:?missing <run_name>: run directory under weights/}
[ $# -eq 2 ] || { echo "expected exactly 2 args, got $#" >&2; exit 1; }
case "${RUN_NAME}" in -*) echo "run_name must not start with '-'" >&2; exit 1;; esac
RUN_DIR="weights/${RUN_NAME}"
STAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "${RUN_DIR}/logs"
CONFIG_SNAP="${RUN_DIR}/logs/config-${STAMP}.json"
cp "${CONFIG_PATH}" "${CONFIG_SNAP}"
cp scripts/train_jobs/sbatch_script.sh "${RUN_DIR}/logs/sbatch_script-${STAMP}.sh"

JOB_ID=$(sbatch --parsable \
  --job-name="${RUN_NAME}" \
  --output="${RUN_DIR}/logs/%x-%j.out" \
  --error="${RUN_DIR}/logs/%x-%j.err" \
  scripts/train_jobs/sbatch_script.sh "${CONFIG_SNAP}" "${RUN_NAME}")

echo "Submitted job ${JOB_ID}: ${RUN_NAME} -> ${RUN_DIR}"
echo "  trains from snapshot ${CONFIG_SNAP}"
