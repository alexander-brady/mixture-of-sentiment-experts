#!/bin/bash
#SBATCH --job-name=cil
#SBATCH --output=logs/ci_l%j.out
#SBATCH --error=logs/cil_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=100G
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpumem:64g


# Load required modules
module load stack/2024-06 gcc/12.2.0 python/3.11.6 cuda/11.3.1 eth_proxy

mkdir -p logs

MODEL=${1:-moe}

# Check if venv exists, create if not
if [ ! -d "$SCRATCH/cil/.venv" ]; then
  python3 -m venv "$SCRATCH/cil/.venv"
  echo "Virtual environment created at $SCRATCH/cil/.venv"
fi

# Activate the virtual environment
source "$SCRATCH/cil/.venv/bin/activate"

echo "Job started at $(date)"

# Create scratch directory for model cache
export HF_HOME="$SCRATCH/cil/model_cache"


# Upgrade pip and install requirements
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

python src/run.py --model "$MODEL"

echo "Job completed at $(date)"

deactivate 