#!/bin/bash
#SBATCH --job-name=mixim-experiments
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=2
#SBATCH --time=05-00:00:00
#SBATCH --mem=200G
#SBATCH --array=1-216%16
#SBATCH --output=Logs/mixim-%A_%a.out
#SBATCH --error=Logs/mixim-%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="/scratch/loq85qob/transition-window"
cd "$ROOT_DIR"

CAMPAIGN_NAME="campaign01"
CAMPAIGN_DIR="$ROOT_DIR/files/$CAMPAIGN_NAME"
MATRIX_FILE="$CAMPAIGN_DIR/experiment_matrix.csv"
REPEATS=10
BIN_SIZE=0.5

mkdir -p "$ROOT_DIR/Logs" "$ROOT_DIR/files"

echo "=== STARTING SBATCH SCRIPT ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-unknown}"
echo "Root dir: $ROOT_DIR"

module load python/3.12 || { echo "ERROR: Python 3.12 failed to load"; exit 1; }

if [ ! -d "$ROOT_DIR/venv_mixim" ]; then
    echo "ERROR: venv_mixim not found in $ROOT_DIR. Run prep.sbatch first."
    exit 1
fi

echo "Activating virtual environment..."
source "$ROOT_DIR/venv_mixim/bin/activate" || { echo "ERROR: venv activation failed"; exit 1; }

if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: $MATRIX_FILE not found. Run prep.sbatch first."
    exit 1
fi

TOTAL_COMBOS="$(python3.12 - "$MATRIX_FILE" <<'PY'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
print(len(df))
PY
)"

if [ "${SLURM_ARRAY_TASK_ID}" -gt "$TOTAL_COMBOS" ]; then
    echo "Array task ${SLURM_ARRAY_TASK_ID} exceeds total combos ${TOTAL_COMBOS}. Exiting cleanly."
    exit 0
fi

TASK_BASE="${SLURM_TMPDIR:-/tmp}"
TASK_DIR="$TASK_BASE/mixim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
TASK_PROJECT="$TASK_DIR/project"
LOCAL_CAMPAIGN_DIR="$TASK_PROJECT/files/$CAMPAIGN_NAME"
COMBO_DIR_NAME="$(printf "combo_%03d" "${SLURM_ARRAY_TASK_ID}")"

echo "Task base: $TASK_BASE"
echo "Task dir:  $TASK_DIR"

rm -rf "$TASK_DIR"
mkdir -p "$TASK_PROJECT"

echo "Copying project to task-local workspace..."
rsync -a \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "venv_mixim" \
  --exclude "files" \
  --exclude "Logs" \
  "$ROOT_DIR/" "$TASK_PROJECT/"

mkdir -p "$LOCAL_CAMPAIGN_DIR"
cp "$MATRIX_FILE" "$LOCAL_CAMPAIGN_DIR/experiment_matrix.csv"

cd "$TASK_PROJECT"

echo "=== STARTING COMBO RUN (LOCAL) ==="
python3.12 run.py \
  --combo-id "${SLURM_ARRAY_TASK_ID}" \
  --repeats "$REPEATS" \
  --bin-size "$BIN_SIZE" \
  --campaign-dir "$LOCAL_CAMPAIGN_DIR"

LOCAL_COMBO_DIR="$LOCAL_CAMPAIGN_DIR/$COMBO_DIR_NAME"
SHARED_COMBO_DIR="$CAMPAIGN_DIR/$COMBO_DIR_NAME"

if [ ! -d "$LOCAL_COMBO_DIR" ]; then
    echo "ERROR: Local combo directory not found: $LOCAL_COMBO_DIR"
    exit 1
fi

mkdir -p "$CAMPAIGN_DIR"

echo "Syncing combo result back to shared campaign folder..."
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$LOCAL_COMBO_DIR/" "$SHARED_COMBO_DIR/"
else
    rm -rf "$SHARED_COMBO_DIR"
    cp -r "$LOCAL_COMBO_DIR" "$SHARED_COMBO_DIR"
fi

echo "Synced: $SHARED_COMBO_DIR"
echo "=== JOB COMPLETED ==="
echo "End time: $(date)"