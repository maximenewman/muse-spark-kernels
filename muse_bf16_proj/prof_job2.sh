#!/bin/bash
# Allocation: export SBATCH_ACCOUNT=<your-allocation> (or sbatch --account=...)
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/%u/okc/logs/%j.out
# submit with: DISABLE_DCGM=1 sbatch prof_job.sh
module load python/3.12 cuda/13.2 >/dev/null 2>&1
source /scratch/$USER/okc/venv/bin/activate
cd /scratch/$USER/okc/muse_bf16_proj
echo "DISABLE_DCGM=$DISABLE_DCGM"
for i in $(seq 60); do [ -z "$(dcgmi -v 2>/dev/null | grep 'Hostengine build info:')" ] && break; sleep 5; done
dcgmi -v 2>/dev/null | grep -q 'Hostengine build info:' && echo "WARNING: DCGM still running"
python -c "import skinny_cuda as s; s.module()"
mkdir -p ncu
prof() {  # name, args...
  name=$1; shift
  ncu --profile-from-start off --set full --import-source yes -f -o ncu/$name python prof.py "$@" > ncu/$name.log 2>&1
  echo "== $name: $(grep check: ncu/$name.log)"
}
prof qkv8_ring_4_6 qkv_gate 8 ring 128 6 4 2
prof qkv8_ring_4_3 qkv_gate 8 ring 128 3 4 2
prof pool8_triton pooler 8 triton '["gemm",32,16,256,1,4,5]'
prof pool8_ring pooler 8 ring 128 3 2 8
python ncu_summary.py ncu/qkv8_triton.ncu-rep ncu/qkv8_ring_4_6.ncu-rep ncu/qkv8_ring_4_3.ncu-rep ncu/pool8_triton.ncu-rep ncu/pool8_ring.ncu-rep
