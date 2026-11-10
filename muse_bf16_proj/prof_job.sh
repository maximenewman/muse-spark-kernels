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
python -c "import skinny_cuda as s; s.module('v2'); s.module('v3')"
mkdir -p ncu
prof() {  # name, args...
  name=$1; shift
  ncu --profile-from-start off --set full --import-source yes -f -o ncu/$name python prof.py "$@" > ncu/$name.log 2>&1
  echo "== $name: $(grep check: ncu/$name.log)"
}
prof qkv1_triton   qkv_gate 1 triton '["gemv",4,1,1024,1,4,1]'
prof qkv1_cuda_v2  qkv_gate 1 cuda 1 2 8 1 --layout v2
prof qkv1_cuda_v3  qkv_gate 1 cuda 1 2 8 1 --layout v3
prof qkv8_triton   qkv_gate 8 triton '["gemm",64,16,128,2,4,5]'
prof qkv8_cuda_v2  qkv_gate 8 cuda 1 1 8 1 --layout v2
prof qkv8_cuda_v3  qkv_gate 8 cuda 1 1 8 1 --layout v3
prof pool1_triton  pooler 1 triton '["gemv",8,1,512,1,8,1]'
prof pool1_cuda_v2 pooler 1 cuda 1 2 8 1 --layout v2
prof pool32_triton pooler 32 triton '["gemm",32,32,256,1,4,5]'
prof pool32_cuda_v2 pooler 32 cuda 1 2 8 1 --layout v2
ls -la ncu
