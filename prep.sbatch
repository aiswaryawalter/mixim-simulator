#!/bin/bash
#SBATCH --job-name=prep-matrix
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=1
#SBATCH --time=00-02:00:00
#SBATCH --mem=8G
#SBATCH --output=Logs/prep-matrix-%j.out
#SBATCH --error=Logs/prep-matrix-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="/scratch/loq85qob/transition-window"
cd "$ROOT_DIR"

CAMPAIGN_NAME="campaign04"
CAMPAIGN_DIR="$ROOT_DIR/files/$CAMPAIGN_NAME"
MATRIX_FILE="$CAMPAIGN_DIR/experiment_matrix.csv"

echo "=== STARTING PREPROCESSING JOB ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Working directory: $ROOT_DIR"
echo "Campaign dir: $CAMPAIGN_DIR"

mkdir -p "$ROOT_DIR/Logs" "$CAMPAIGN_DIR"

module load python/3.12 || { echo "ERROR: Python 3.12 failed to load"; exit 1; }

echo "Python version: $(python3.12 --version)"
echo "Python path: $(which python3.12)"

if [ ! -d "venv_mixim" ]; then
    echo "Creating Python virtual environment..."
    python3.12 -m venv venv_mixim || { echo "ERROR: venv creation failed"; exit 1; }
fi

echo "Activating virtual environment..."
source venv_mixim/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }

echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Writing experiment matrix..."
python3.12 run.py --write-matrix --campaign-dir "$CAMPAIGN_DIR"

if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: Matrix file not created: $MATRIX_FILE"
    exit 1
fi

TOTAL_COMBOS="$(python3.12 - "$MATRIX_FILE" <<'PY'
import sys
import pandas as pd
df = pd.read_csv(sys.argv[1])
print(len(df))
PY
)"

echo "Matrix written: $MATRIX_FILE"
echo "Total combos: $TOTAL_COMBOS"
echo "First 5 rows:"
head -n 6 "$MATRIX_FILE"

echo "=== PREPROCESSING COMPLETED ==="
echo "End time: $(date)"