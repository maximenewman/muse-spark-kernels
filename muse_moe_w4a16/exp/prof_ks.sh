#!/bin/bash
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/okc/logs/%j.out
module load python/3.12 cuda/13.2 >/dev/null 2>&1
source /scratch/$USER/okc/venv-sgl/bin/activate
cd /scratch/$USER/okc/muse_moe_w4a16
mkdir -p ncu
for c in "8,4,4,1,1" "8,4,4,2,2" "8,4,4,4,4" "8,2,4,4,4"; do
  CFG=$c ncu --profile-from-start off --set full -f -o ncu/ks_${c//,/_} python prof.py 1 > /dev/null 2>&1
done
python ncu_summary.py ncu/ks_*.ncu-rep
