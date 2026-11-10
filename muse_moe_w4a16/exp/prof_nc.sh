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
for c in "8,4,4,1,1" "8,8,4,1,1" "8,4,4,4,4"; do
  NOCOMPUTE=1 CFG=$c ncu --profile-from-start off --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats -f -o ncu/nc_${c//,/_} python prof.py ${M:-1} > /dev/null 2>&1
done
python ncu_summary.py ncu/nc_*.ncu-rep
