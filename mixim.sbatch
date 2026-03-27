#!/bin/bash
#SBATCH --job-name=arrivaltime
#SBATCH --error=Logs/mixim-%j.err
#SBATCH --partition=haswell-256
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=50G

# Force output to flush immediately
export PYTHONUNBUFFERED=1

# Exit on any error and log it
set -e

# Add system monitoring
echo "=== STARTING SBATCH SCRIPT ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Available memory: $(free -h)"

module load python/3.12 || { echo "ERROR: Python 3.12 failed to load"; exit 1; }
echo "Python loaded successfully"

mkdir -p Logs

echo "Creating Python virtual environment..."
python3.12 -m venv venv_mixim || { echo "ERROR: venv creation failed"; exit 1; }

echo "Activating virtual environment..."
source venv_mixim/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }

echo "Python version: $(python3.12 --version)"
echo "Python path: $(which python3.12)"

which python
python --version

echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt
# export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH

echo "Starting job at $(date)"
echo "Available memory: $(free -h)"

echo "=== TESTING IMPORTS ==="
python3.12 -c "from main import main; print('Import successful')" || {
    echo "ERROR: Failed to import main"
    exit 1
}

echo "=== STARTING JOB ==="
python3.12 plot.py 2>&1 | tee Logs/mixim-detailed-${SLURM_JOB_ID}.log

EXIT_CODE=$?
echo "=== JOB COMPLETED with exit code: $EXIT_CODE ===" 
echo "End time: $(date)"
echo "Final memory usage: $(free -h)"

exit $EXIT_CODE
