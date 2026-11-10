#!/bin/bash
# Allocation: export SBATCH_ACCOUNT=<your-allocation> (or sbatch --account=...)
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/okc/logs/%j.out
# usage: sbatch job.sh python bench.py [args]
module load python/3.12 cuda/13.2 >/dev/null 2>&1
source /scratch/$USER/okc/venv-sgl/bin/activate
cd /scratch/$USER/okc/muse_moe_w4a16
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv,noheader
"$@"
