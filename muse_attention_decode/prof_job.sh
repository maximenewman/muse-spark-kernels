#!/bin/bash
# Allocation: export SBATCH_ACCOUNT=<your-allocation> (or sbatch --account=...)
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/okc/logs/%j.out
module load python/3.12 cuda/13.2 >/dev/null 2>&1
source /scratch/$USER/okc/venv-sgl/bin/activate
cd /scratch/$USER/okc/muse_attention_decode
for i in $(seq 60); do [ -z "$(dcgmi -v 2>/dev/null | grep 'Hostengine build info:')" ] && break; sleep 5; done
mkdir -p ncu
for spec in "decode_1 full" "verify_32 full"; do
  set -- $spec; name=${1}_$2
  ncu --profile-from-start off --set full -f -o ncu/$name python prof.py $1 $2 > ncu/$name.log 2>&1

done
python ncu_summary.py ncu/*.ncu-rep
