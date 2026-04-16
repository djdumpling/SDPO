#!/bin/bash
#SBATCH --job-name=sdpo-qwen30b-litbench
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/aw2295/scratch_pi_btk22/aw2295/sdpo/rubric-policy-optimization}"
CONDA_ENV="${CONDA_ENV:-sdpo}"

source "$HOME/.bashrc"
conda activate "$CONDA_ENV"

# Judge credentials (OPENAI_API_KEY, OPENAI_BASE_URL). Keep them out of git.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi

cd "$REPO_ROOT/SDPO"
mkdir -p logs

nvidia-smi

bash ./run_sdpo_qwen30b_thinking.sh "${SLURM_JOB_ID:-manual}"
