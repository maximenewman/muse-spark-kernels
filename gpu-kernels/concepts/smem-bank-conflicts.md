---
title: Shared Memory Bank Conflicts
type: concept
tags: [gpu, smem, bank-conflicts, hopper, swizzle]
source_url:
  - https://modal.com/gpu-glossary/perf/bank-conflict
  - https://feldmann.nyc/blog/smem-microbenchmarks
  - https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
---
# Shared Memory Bank Conflicts

**TL;DR**
- Shared memory = 32 banks × 4 B; `bank = (byte_addr / 4) % 32`; addresses 128 B apart share a bank (Modal, Feldmann).
- Within one warp request, N distinct addresses in the same bank → N serialized passes. Same address → broadcast (free); arbitrary multicast is free too (Feldmann).
- Conflicts in different warps don't overlap – they all serialize (Feldmann, A100).
- 16 B/lane vector loads (`ld.shared.v4`, `LDS.128`) run at full rate: hardware staggers lane bank order; a warp's 512 B takes ~4 bank-cycles, which is optimal, not a conflict.
- Classic fixes: padding (+1 word per row), XOR swizzling ([[swizzling]]), or transposing the access.

## Key ideas
- Peak smem bandwidth per SM = 32 banks × 4 B = 128 B/cycle (inference from the bank model).
- Column access of a row-major `float[32][32]` (`data[tid*32]`) → all lanes bank 0 → 32-way conflict; Modal: latency "from roughly 10 cycles to hundreds".
- Transpose through smem is the textbook case: one of (read rows, write columns) is strided by the row pitch. Veitner: naive TMA transpose 875 GB/s → 128B-swizzled 1252 GB/s.
- Tensor-core operand paths (ldmatrix / wgmma smem descriptors / TMA) work in 16 B chunks; the relevant unit is "which group of 4 banks does each 16-B chunk hit" – the CUDA guide: swizzle patterns map "16-byte chunks along the swizzle width to subgroups of four banks". 8 chunks × 16 B = 128 B = one full sweep of 32 banks.
- Padding vs swizzle: padding wastes smem and breaks TMA/wgmma layouts (they require specific swizzle modes, not padded pitches); swizzle is free in capacity and is what TMA writes natively.

## Numbers worth remembering
- 32 banks, 4 B each (Modal; Feldmann).
- A100: conflict-free ≈ 1 load/bank/cycle; 32-way conflict ≈ 32× slower (0.57 ms → 18.2 ms per 100k iters) (Feldmann).
- v4 loads: 4× data in 4× time – full bandwidth (Feldmann).

## Code patterns
```cpp
__shared__ float tile[32][32];
float a = tile[threadIdx.y][threadIdx.x];   // row read: lane i -> bank i, conflict-free
float b = tile[threadIdx.x][threadIdx.y];   // column read: stride 128 B -> 32-way conflict
__shared__ float tileP[32][33];             // padding: row pitch 33 words -> lane i -> bank (i*33)%32 = i
float c = tileP[threadIdx.x][threadIdx.y];  // conflict-free, costs 32 extra words
```
XOR swizzle alternative (16-B chunk granularity, 128 B rows): `chunk' = chunk ^ (row & 7)`; see [[swizzling]].

## Pitfalls
- Counting "conflicts" in Nsight Compute: v4/v2 loads show multiple wavefronts per request even when optimal; compare against ideal wavefronts (bytes/128 per request), not zero.
- 8-byte and 16-byte accesses with a stride that is a multiple of 128 B still conflict at chunk level – e.g. all lanes reading chunk 0 of different 128-B rows.
- A swizzle that is right for the TMA write side must match the read side (same mode, buffer aligned to the swizzle period).

## Relevance to the serving track
- (inference) In a weight-streaming GEMV/skinny GEMM, smem is rarely the limiter: per-SM HBM share at 4.8 TB/s ÷ 132 SMs is ~36 GB/s, far below the per-SM smem rate. Bank conflicts matter only if they add latency to the critical path (e.g. ldmatrix from a non-swizzled TMA tile delaying the MMA that frees a pipeline stage) or if many warps conflict at once.
- (inference) Broadcasting x (activations) from smem to all lanes is free; keep x in smem, stream W.
- (inference) If W tiles are TMA-loaded, choose `CU_TENSOR_MAP_SWIZZLE_128B` with a 128-B (64 BF16) inner box; then wgmma/ldmatrix reads are conflict-free by construction.

## Related
[[swizzling]] · [[feldmann-smem-microbenchmarks]] · [[veitner-hopper-transpose]] · [[tma]] · [[wgmma]] · [[mma-ptx]] · [[gpu-memory-hierarchy]]
