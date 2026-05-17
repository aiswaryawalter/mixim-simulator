#!/bin/bash
#SBATCH --job-name=mixim-matrix
#SBATCH --partition=haswell-256
#SBATCH --cpus-per-task=2
#SBATCH --time=00-02:00:00
#SBATCH --mem=8G
#SBATCH --output=Logs/prep-matrix-%j.out
#SBATCH --error=Logs/prep-matrix-%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT_DIR="/scratch/loq85qob/instant-corruption-0"
cd "$ROOT_DIR"

echo "=== STARTING PREPROCESSING JOB ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start time: $(date)"
echo "Working directory: $ROOT_DIR"

module load python/3.12 || { echo "ERROR: Python 3.12 failed to load"; exit 1; }

echo "Python loaded successfully"
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

CAMPAIGN_DIR="$ROOT_DIR/files/rho_sweep"
mkdir -p "$CAMPAIGN_DIR"

echo "Writing experiment matrix to $CAMPAIGN_DIR"
python3.12 run.py --write-matrix --campaign-dir "$CAMPAIGN_DIR"

echo "Matrix written successfully:"
ls -l "$CAMPAIGN_DIR/experiment_matrix.csv"

echo "=== PREPROCESSING COMPLETED ==="
echo "End time: $(date)"