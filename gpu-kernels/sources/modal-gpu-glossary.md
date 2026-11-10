---
title: Modal GPU Glossary (digest)
type: source
tags: [gpu, glossary, tensor-core, memory-hierarchy, roofline, littles-law, occupancy]
source_url:
  - https://modal.com/gpu-glossary
  - https://modal.com/gpu-glossary/device-hardware/tensor-core
---
# Modal GPU Glossary (digest)

**TL;DR**
- Short, well-linked definitions of GPU hardware/software/perf terms. Best pages for memory-bound work: Little's law, latency hiding, memory-bound, roofline, arithmetic intensity, memory bandwidth (has a ridge-point table), occupancy (H100 limits + worked example), bank conflict.
- Tensor Cores: matrix-per-instruction cores (since Volta), ~100× CUDA-core FLOPS; 4 per SM on H100 SXM5 (one per warp scheduler). Peak on Hopper/Blackwell needs PTX intrinsics for compute AND memory (wgmma/TMA), not plain CUDA C++.
- Ridge points (dense BF16): H100 SXM 989 TFLOPs / 3.35 TB/s = 295 FLOP/B. Any GEMM with arithmetic intensity well below that is memory-bound.
- Little's law: concurrency = latency × throughput. Example: 400-cycle latency at 1 op/cycle → 400 ops in flight.
- Occupancy is not a goal in itself; Hopper GEMMs often run at single-digit occupancy.

## Key ideas
- **Tensor Core** (`device-hardware/tensor-core`): `HMMA16.16816.F32` = D=AB+C, m16n8k16, FP16 in / FP32 accum, 2048 MACs per instruction, 64 MACs/thread across a warp. `wmma::mma_sync` 16×16 compiles to 4× `HMMA.1688.F32`. Internals undocumented (systolic array assumption unproven). Treat Tensor Cores as CISC.
- **Memory hierarchy** (`device-software/memory-hierarchy`): registers (thread, in SM register file) / shared memory (block, physically in L1 data cache) / global (grid, in GPU RAM); "managed by the programmer". No numbers on that page.
- **L1 data cache**: 256 KiB per SM on H100; ×132 SMs = 33 MiB total (page's bit-count figure is inconsistent). ~an order of magnitude slower than cores. No standalone L2 page exists.
- **Register file**: ~order of magnitude faster than L1; 32-bit registers; register pressure lowers occupancy.
- **GPU RAM**: H100 80 GiB; on-interposer HBM for bandwidth.
- **Memory bandwidth** page: table of TFLOPs, TB/s, ridge point (below). B200 HBM3e 8 TB/s.
- **Little's law**: concurrency = latency × throughput; Volkov thesis §4.3: ~30 warps needed to hide pure memory latency vs ~24 for arithmetic (older GPUs).
- **Latency hiding**: SASS example `LDG` 400 cy + IMUL 6 + IADD 4 + IMUL 6 = 416 cy → 416 threads = 13 warps in flight to issue once/cycle.
- **Memory-bound**: decode is memory-bound because weights are reloaded per token; e.g. 500B params × 2 B = 1 TB at 10 TB/s = 100 ms floor; batching raises FLOPs per weight byte "without adding latency" until compute-bound.
- **Roofline**: x = arithmetic intensity, y = perf; ridge point = min intensity to escape memory bound; ignores latency (only bandwidths/throughputs). Nsight Compute draws it.
- **Arithmetic intensity**: SAXPY 1/6; SGEMM 2N³ / 16N² = N/8 FLOP/B.
- **Occupancy**: H100: 64 warps/SM max, 32 blocks/SM, 65,536 regs, 228 KB smem. Worked example: 32 thr/block, 8 regs, 12 KB smem → smem limits to 19 blocks. More occupancy beyond what latency hiding needs can hurt.
- **Bank conflict**: 32 banks × 4 B, addresses 128 B apart share a bank; `data[tid*32]` → 32-way, "~10 cycles to hundreds"; same address = broadcast.

## Numbers worth remembering
From `perf/memory-bandwidth` and `perf/arithmetic-intensity` (dense numbers as given by Modal):
| System | TFLOP/s | TB/s | Ridge FLOP/B |
|---|---|---|---|
| A100 80GB SXM BF16 / HBM2e | 312 | 2 | 156 |
| H100 SXM BF16 / HBM3 | 989 | 3.35 | 295 |
| H100 SXM FP8 / HBM3 | 1979 | 3.35 | 592 |
| B200 BF16 / HBM3e | 2250 | 8 | 281 |
| B200 FP8 | 4500 | 8 | 562 |
| B200 FP4 | 9000 | 8 | 1125 |
No H200 row. (inference) H200 SXM: NVIDIA lists 4.8 TB/s and 1,979 BF16 TFLOPS *with sparsity* → ~989 dense → ridge ≈ 206 FLOP/B (see [[roofline-and-arithmetic-intensity]]).

H100 per-SM limits (occupancy page): 64 warps, 32 blocks, 65,536 × 32-bit registers, 228 KB smem. L1: 256 KiB/SM.

## Code patterns
```cpp
__shared__ float data[1024];
float v = data[threadIdx.x];        // consecutive 4B words -> 32 banks, 1 transaction
float w = data[threadIdx.x * 32];   // stride 128 B -> every lane bank 0 -> 32-way conflict
```

## Pitfalls
- Glossary numbers are spec-sheet/peak; H100 bandwidth appears as 3.35 TB/s here vs "up to 3 TB/s" in NVIDIA's Hopper tuning guide (see [[gpu-memory-hierarchy]]).
- Roofline ignores latency – useless on its own for 2–17 µs kernels where ramp-up and latency dominate.

## Relevance to the serving track
- (inference) For `muse_bf16_proj`, Little's law + latency hiding pages are the core model: the kernel must keep enough loads outstanding to cover HBM latency on every SM from the first microsecond. See [[memory-bandwidth-and-latency]].
- (inference) Occupancy page's "single-digit occupancy is fine" applies: a TMA-pipelined producer/consumer kernel with 1 CTA/SM can saturate HBM if the pipeline holds enough bytes.

## Most useful glossary URLs
Perf:
- https://modal.com/gpu-glossary/perf/littles-law
- https://modal.com/gpu-glossary/perf/latency-hiding
- https://modal.com/gpu-glossary/perf/memory-bound
- https://modal.com/gpu-glossary/perf/memory-bandwidth
- https://modal.com/gpu-glossary/perf/roofline-model
- https://modal.com/gpu-glossary/perf/arithmetic-intensity
- https://modal.com/gpu-glossary/perf/occupancy
- https://modal.com/gpu-glossary/perf/bank-conflict
- https://modal.com/gpu-glossary/perf/memory-coalescing
- https://modal.com/gpu-glossary/perf/register-pressure
- https://modal.com/gpu-glossary/perf/scoreboard-stall
- https://modal.com/gpu-glossary/perf/issue-efficiency
- https://modal.com/gpu-glossary/perf/warp-execution-state
- https://modal.com/gpu-glossary/perf/overhead
- https://modal.com/gpu-glossary/perf/peak-rate
Hardware:
- https://modal.com/gpu-glossary/device-hardware/tensor-core
- https://modal.com/gpu-glossary/device-hardware/tensor-memory-accelerator
- https://modal.com/gpu-glossary/device-hardware/warp-scheduler
- https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor
- https://modal.com/gpu-glossary/device-hardware/load-store-unit
- https://modal.com/gpu-glossary/device-hardware/register-file
- https://modal.com/gpu-glossary/device-hardware/l1-data-cache
- https://modal.com/gpu-glossary/device-hardware/gpu-ram
Software:
- https://modal.com/gpu-glossary/device-software/memory-hierarchy
- https://modal.com/gpu-glossary/device-software/shared-memory
- https://modal.com/gpu-glossary/device-software/registers
- https://modal.com/gpu-glossary/device-software/global-memory
- https://modal.com/gpu-glossary/device-software/warpgroup
- https://modal.com/gpu-glossary/host-software/cuda-graph
(Not fetched in full: warp-scheduler, shared-memory, registers, TMA, coalescing pages – only listed.)

## Related
[[gpu-memory-hierarchy]] · [[memory-bandwidth-and-latency]] · [[roofline-and-arithmetic-intensity]] · [[smem-bank-conflicts]] · [[wgmma]] · [[tma]] · [[cuda-graphs]]
