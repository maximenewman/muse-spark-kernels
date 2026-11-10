---
title: "A history of NVidia Stream Multiprocessor (Fabien Sanglard)"
type: source
tags: [gpu, history, architecture, sm, simt, warps]
source_url: https://fabiensanglard.net/cuda/
---
# A history of NVIDIA SMs (Fabien Sanglard)

**TL;DR**
- Before 2006, GPUs had fixed pipeline stages (e.g. G71: 8 vertex, 24 fragment, 16 merge units) and suffered load imbalance.
- 2006 Tesla/G80 (8800 GTX) introduced unified SMs, SIMT execution in **32-thread warps**, and CUDA C (a C99 superset): 128 cores, 345.6 GFLOPS.
- Each generation since has mostly reshaped the SM (scheduler count, cores/SM) and ridden process shrinks. Turing (2018) re-added specialized units (Tensor and RT cores).
- Volta/Turing **Independent Thread Scheduling** gives each thread its own PC, so warp lanes may diverge and not reconverge implicitly.

## Key ideas (timeline)
| Year | Arch / chip | Notable SM change | Headline |
|---|---|---|---|
| 2006 | Tesla G80 (90 nm) | Unified SM, SIMT warps of 32, 2 SFUs at 1/4 rate, no FP64 | 128 cores, 345.6 GFLOPS |
| 2010 | Fermi GF100 (40 nm) | 1 warp-instr/clk scheduler, FMA, semi-HW FP64, C++ in CUDA, GPCs | 512 cores, 1,345 GFLOPS |
| 2012 | Kepler GK104 (28 nm) | SMX: 4 schedulers, dual dispatch, 192 cores/SMX | 1,536 cores, 3,250 GFLOPS |
| 2014 | Maxwell GM200 (28 nm) | SMM 128 cores (warp-aligned partitions), power efficiency | 980 Ti: 3,072 cores, 6,060 GFLOPS |
| 2016 | Pascal GP102 (16 nm) | SM ≈ Maxwell, more SMs via shrink, +43% bandwidth | 1080 Ti: 3,584 cores, 11,340 GFLOPS |
| 2018 | Turing TU102 (12 nm) | Tensor + RT cores, concurrent INT/FP issue, ITS | 2080 Ti: 68 SMs, 13.45 TFLOPS |

## Pitfalls
- The article stops at Turing (written pre-Ampere). Nothing on async copy (Ampere), TMA/wgmma/clusters (Hopper), or tcgen05/TMEM (Blackwell). See [[tma]], [[wgmma]].
- ITS means warp-synchronous code without `__syncwarp()`/`*_sync` intrinsics is unsafe on Volta+.

## Relevance to the serving track
- Background only. The lasting concepts are the warp as the SIMT unit (shuffles, warp-level reductions in GEMV split-K) and the trend of throughput growing faster than memory bandwidth, which is why decode GEMMs are memory-bound. (inference)

## Related
[[gtc25-s72449-modern-cuda]] · [[wgmma]] · [[tma]] · [[gemv-and-skinny-gemm]]
