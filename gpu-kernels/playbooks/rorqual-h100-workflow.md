---
title: Running and tuning kernels on Rorqual H100s (local judge stand-in)
type: playbook
tags: [workflow, rorqual, h100, slurm, benchmarking, tuning, triton, muse]
source_url:
  - ../muse_bf16_proj/bench.py
  - ../muse_bf16_proj/tune.py
  - ../muse_bf16_proj/job.sh
  - https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm
---
# Running and tuning kernels on Rorqual H100s

**TL;DR**
- The okc kernel track is closed (`okc check` returns "the kernel track has closed"), so kernel problems are timed locally with `bench.py`, a stand-in for the judge.
- GPUs come from the Alliance cluster **Rorqual** (H100-80GB SXM, 3.35 TB/s) on a Slurm allocation (`$SBATCH_ACCOUNT`).
- **H100 numbers are estimates of the H200 board.** Bandwidth-bound ratios against cuBLAS carry over roughly, but the best configs can shift.

## Key ideas
### Access
- `ssh rorqual` (cluster user `$USER`). The ssh config uses ControlMaster, so a person runs `ssh -fN rorqual` once (Duo MFA) and agents reuse the socket for 12 h.
- Work dir: `/scratch/$USER/okc/<problem>/`. The venv is `/scratch/$USER/okc/venv`: python 3.12, torch 2.14.1 + CUDA 13.2, triton 3.6, from the Alliance wheelhouse (`pip install --no-index`).
- Sync with `rsync -az --exclude exp --exclude profiles ./ rorqual:/scratch/$USER/okc/<problem>/`.

### Jobs (`job.sh`)
- `#SBATCH --account=$SBATCH_ACCOUNT --gpus=h100:1 --cpus-per-task=16 --mem=64G`. Modules: `python/3.12 cuda/13.2`.
- Logs go to `/scratch/$USER/okc/logs/<jobid>.out`. Submit as `sbatch [--time=..] job.sh python bench.py ...`.
- Queue waits ranged from seconds to ~30 min. A 15–30 min `--time` schedules faster.
- **Cancel by explicit job ID.** `scancel -u` gets blocked by the agent's safety classifier.

### bench.py (the judge stand-in for muse_bf16_proj)
- Runs the 9 judge tests and all timed shapes through `reference.check_implementation`.
- Each shape is captured as a CUDA graph for both baseline and submission. Every replay is preceded by a 512 MB read (cold L2). Baseline and submission alternate in AB/BA pairs, and the speedup is the median of paired ratios.
- It subtracts a measured fixed replay cost (~4.8 µs: a graph holding one trivial kernel; the judge says ~4 µs). Load time = Σ weight × time, and the problem score is the geomean over loads.
- Baseline: SGLang's `_bf16_gemm_dispatch_impl` is `F.linear` (cuBLAS) on H200 with default flags. bench.py stubs the sglang import, so sglang isn't installed.

### tune.py
- Sweeps a focused config space per (layer, m), all under graph replay with a cold L2. It re-times the top 5 at 5× the reps and writes `tuned*.json` plus a `CONFIGS = {...}` block.
- **Parallel precompile:** a spawn-pool of 62 workers runs each config once to fill Triton's on-disk cache. Compile time for ~400 configs dropped from **~4 s/config serial (>30 min per shape) to ~40 s total**. Timing then runs alone on the GPU.
- Keep the search space to ~30–120 configs per shape. A naive product space (~1,300+ configs) didn't finish a single shape in 30 min.

### Running real SGLang kernels as the baseline (muse_attention_decode and later problems)
- `okc serve init sglang-pinned` fetches SGLang at `a55ab7d5`. It pins `torch==2.14.1` (the same as the Alliance wheel) and `sglang-kernel==0.4.9` (FA3 lives in `sgl_kernel`).
- **The Alliance pip only sees its wheelhouse,** which has sglang-kernel 0.4.4, and its Python **rejects manylinux wheels**. Workaround, in a separate venv `/scratch/$USER/okc/venv-sgl`:
  1. `curl` the wheel URLs from `https://pypi.org/pypi/<pkg>/<ver>/json`.
  2. `python -m zipfile -e <whl> $SITE_PACKAGES`.
  3. `setrpaths.sh --path $SITE_PACKAGES/<pkg>`.
  4. `PIP_CONFIG_FILE=/dev/null pip install --no-deps --no-build-isolation -e sglang-python`.
- The import chain needed 28 small packages (orjson, pydantic, aiohttp, IPython, …). They were installed with a loop that parses `No module named X` and tries `pip install --no-deps --no-index X`, then PyPI. The user approved the PyPI installs on 2026-11-09.
- The attention bench keeps two independent input sets per shape (~700 MB each at B=32) and alternates their graphs instead of copying fresh inputs.

### Profiling with Nsight Compute
- `ncu`/`nsys` come with `module load cuda/13.2`.
- Submit with `DISABLE_DCGM=1 sbatch prof_job.sh`. The job waits while `dcgmi -v` shows a host engine. In our runs it still printed "DCGM still running" after 5 min, but `--set full` collected all metrics anyway.
- `prof.py <layer> <m> <triton|cuda|ring> <cfg>` warms up, checks correctness, then runs one call between `cudaProfilerStart/Stop`. Use `ncu --profile-from-start off`.
- `ncu_summary.py *.ncu-rep` prints time, warps/eligible per scheduler, regs, grid and the top-5 `smsp__average_warps_issue_stalled_*` reasons. Run it on the login node; no GPU needed.
- Useful raw metrics:
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (sector efficiency)
  - `smsp__inst_executed.sum` (instruction bloat)
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
  - `smsp__inst_executed_op_ldgsts.sum` (cp.async count)
- Ten `--set full` captures take ~9 min in total. ncu runs at base clocks, so compare kernels against each other, not against bench µs.

## Numbers worth remembering
- H100 cuBLAS (SGLang baseline) under this harness, µs: qkv_gate 15.7–16.7, o 6.4–8.4, pooler 25.1–30.7, unpooler-full 23–27, unpooler-slice 6.4–6.6, router 9.8 (m=1), 25.0 (m=8), 14.8–17.4 (m ≥ 16).
- H100 floors (W bytes / 3.35 TB/s): qkv 11.3, o 5.0, pooler/unpooler 20.0, slice/router 2.5 µs.
- Run-to-run noise on identical cuBLAS shapes: ±1–2%.

## Pitfalls
- Keep the same Triton argument form (runtime vs `tl.constexpr`) between tuning and the final run. The best config changes between them (see [[findings-log]]).
- `load_inline` CUDA extensions take ~60 s to build on a 16-CPU node. Build once before spawning precompile workers, or they race on the build dir.
- `torch.sum(buf, out=scalar)` needs `dim=0` in torch 2.14.

## Relevance to the serving track
- The same access, venv and Slurm pattern works for serving experiments, but serving runs need 8×H200, which this allocation doesn't have. Use `okc serve check` for those.

## Related
[[benchmarking-kernels]] · [[cuda-graphs]] · [[triton]] · [[findings-log]] · [[gemv-and-skinny-gemm]] · [[register-direct-weight-streaming]]
