---
title: gpu-kernels — index
type: index
tags: [index]
---
# gpu-kernels — index

A knowledge base for writing fast GPU kernels on Hopper (H100/H200), aimed at the **Muse Spark Open Kernel Challenge serving track**. The seed was [sfu-arch/Hidden-Treasures/LLMs](https://github.com/sfu-arch/Hidden-Treasures/tree/master/LLMs) (see [[hidden-treasures-llms]]); every link in its reading list is digested below. Agents: read `AGENTS.md` first.

## Start here (challenge)
- [[serving-track-playbook]]: rules, scoring ladder, stock launch config, decode-time budget, ranked levers
- [[muse-spark-open-model]]: architecture, per-GPU shapes, one layer's math, the kernels stock SGLang uses
- [[findings-log]]: measured results (append here)
- [[rorqual-h100-workflow]]: how to bench/tune on Rorqual H100s now the okc kernel judge is closed (bench.py, tune.py, Slurm, parallel precompile)
- `raw/okc/`: primary challenge docs (`llms.txt`, `serving.json`, `problem-*.txt` specs for all 5 kernel problems)

## Small-m / memory-bound core (read these for decode & verify)
- [[gemv-and-skinny-gemm]]: why m ≤ 256 projections are HBM-bound; design space, per-layer floors, levers
- [[roofline-and-arithmetic-intensity]]: intensity ≈ m FLOP/B; where m = 128/256 stops being free
- [[memory-bandwidth-and-latency]]: Little's law; more than 32 KiB in flight per SM on Hopper
- [[split-k-and-stream-k]]: filling 132 SMs at small n; self-resetting last-CTA reduction
- [[split-kv-decode-attention]]: paged FP8 decode/verify attention in Triton (block-diagonal GQA, vectorized merge, FP8 QK); 1.44× over SGLang FA3 on H100
- [[register-direct-weight-streaming]]: W straight into mma.sync fragments via a k-permutation; measured slower than tuned Triton on H100 (negative result)
- [[speculative-decoding-shapes]]: verify blocks give m = 8 × batch
- [[weight-only-quantization]]: int4/int8/NVFP4 dequant in kernel (no FP4 tensor cores on H200)
- [[cuda-graphs]] · [[pdl]] · [[benchmarking-kernels]]

## Hopper hardware features
- [[tma]] · [[mbarrier]] · [[wgmma]] · [[warp-specialization]] · [[thread-block-clusters]] · [[persistent-kernels-and-tile-scheduling]]
- [[swizzling]] · [[smem-bank-conflicts]] · [[gpu-memory-hierarchy]]
- [[tensor-core-generations]] · [[tcgen05-tmem]] (Blackwell only; for contrast)

## Classic GEMM building blocks
- [[gemm-tiling-hierarchy]] · [[register-tiling]] · [[mma-ptx]] (mma.sync / ldmatrix) · [[cp-async]] · [[cute-layouts]]

## Tools / DSLs
- [[triton]]: when to pick Triton vs CUDA vs CuTe at small m; split-K variants; autotune

## Sources (one digest per article)
| Note | What |
|---|---|
| [[hidden-treasures-llms]] | The seed repo and its reading list mapped to notes |
| [[hidden-treasures-3-gemm-code]] | Kernels 1→6 of the local HGEMM code, diff by diff (`raw/hidden-treasures-llms/3-GEMM`) |
| [[alexarmbr-tensor-core-hgemm]] | Tensor-core HGEMM from scratch on T4, ~96% of cuBLAS |
| [[siboehm-sgemm-worklog]] | 10-step FP32 SGEMM worklog, up to 93.7% of cuBLAS |
| [[h100-gemm-worklog-cudaforfun]] | H100 BF16 GEMM, 32 → 764 TFLOPs (beats cuBLAS's 716) |
| [[colfax-blackwell-tmem-gemm]] | Blackwell tcgen05 + TMEM GEMM in CUTLASS |
| [[semianalysis-tensor-core-evolution]] | Tensor cores from Volta to Blackwell (partly paywalled) |
| [[bruce-lee-mma-ptx]] | WMMA vs MMA PTX HGEMM ladders, `ptx.h` macros |
| [[yangyifan-cute-tma]] · [[yangyifan-pdl]] · [[yangyifan-cute-layouts]] · [[yangyifan-register-tiling]] | Yang YiFan's Hopper / CuTe blog posts |
| [[veitner-hopper-transpose]] | Fast transpose on Hopper (TMA + swizzle) |
| [[feldmann-smem-microbenchmarks]] | Shared-memory bank-conflict microbenchmarks |
| [[gtc25-s72683-memory]] | GTC25: max bandwidth, bytes in flight, TMA vs cp.async |
| [[modal-gpu-glossary]] | Modal GPU glossary digest + URL list |
| [[pytorch-gpt-fast-accelerating-genai-2]] | gpt-fast: compile, CUDA graphs, int4/int8, spec decoding |
| [[triton-docs]] | Triton docs: API, tutorials, `do_bench` internals |
| [[modular-triton-and-python-edsls]] | Modular's critique of Python eDSLs (opinion piece) |
| [[gtc25-s72449-modern-cuda]] | GTC25: "1,001 Ways to Write CUDA Kernels in Python" |
| [[sanglard-gpu-history]] | GPU history G80 → Turing |

## Known gaps / to do
- No notes yet on: NVFP4 Marlin MoE internals, FA3 decode internals, SGLang's code layout at `a55ab7d5`, all-reduce fusion, the DFlash drafter. These are the next sources to add for the serving track (run `okc serve init` to get the pinned SGLang source).
- Not sourced anywhere in the vault: H200 SM count and L2 size, and dense BF16 TFLOPs (only spec-sheet / inference values).
- Unread: Bruce-Lee's Medium MMA article (HTTP 403), SemiAnalysis's paywalled second half, and the GTC talk videos (only secondary notes were used).
