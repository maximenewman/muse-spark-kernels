---
title: "siboehm: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance (SGEMM worklog)"
type: source
tags: [gpu, gemm, sgemm, cuda, tiling, fp32, worklog]
source_url: https://siboehm.com/articles/22/CUDA-MMM
---
# siboehm SGEMM worklog

**TL;DR**
- FP32 SGEMM on **CUDA cores (FFMA), no tensor cores**, built up in 10 kernels from naive (1.3% of cuBLAS) to warptiling (93.7%). Mostly on an RTX A6000, 4092² matrices (the post's own number; possibly meant 4096).
- The jumps that matter: coalescing (6.4×), then **register tiling**: 1D blocktiling ~2.8×, 2D blocktiling ~1.9×. Each thread computes a TM×TN outer product from register-cached slices of A and B.
- Later steps: vectorized `float4` LDS/LDG (+14%), an autotuning sweep over BM/BN/BK/TM/TN (+8%), and a warp-level tile between block and thread tiles (+10%).
- **Directly relevant to small m:** cuBLAS itself switches kernels by size, and **at 256³ it uses split-K plus a separate `splitKreduce_kernel`**. Small problems are where a hand kernel is weakest against cuBLAS.
- This is the canonical CUDA-core GEMM reference. For BF16 on H100/H200, the tensor-core path is in [[alexarmbr-tensor-core-hgemm]] and [[mma-ptx]].

## Key ideas
| # | Kernel | What changes | GFLOP/s | % cuBLAS |
|---|---|---|---|---|
| 1 | Naive | One thread per C element, grid `CEIL_DIV(M,32)×CEIL_DIV(N,32)`, 32×32 blocks | 309.0 | 1.3% |
| 2 | GMEM coalescing | Remap so consecutive `threadIdx.x` reads consecutive addresses (`threadIdx.x / BLOCKSIZE` → row, `% BLOCKSIZE` → col) | 1986.5 | 8.5% |
| 3 | SMEM cache-blocking | 32×32 tiles of A and B staged in SMEM (8 KB), partial sums over K | 2980.3 | 12.8% |
| 4 | 1D blocktiling | Each thread computes TM=8 outputs. BM=BN=64, BK=8 | 8474.7 | 36.5% |
| 5 | 2D blocktiling | Each thread computes TM×TN = 8×8 = 64 outputs, as an outer product of `regM`×`regN` | 15971.7 | 68.7% |
| 6 | Vectorize | Store A **transposed** in SMEM so the inner-loop As reads become `LDS.128`. `float4` GMEM loads/stores (`LDG.E.128`/`STG.E.128`) | 18237.3 | 78.4% |
| 7–8 | Bank-conflict experiments | Slower, dropped | – | – |
| 9 | Autotuning | Sweep of ~400 configs of BM, BN, BK, TM, TN | 19721.0 | 84.8% |
| 10 | Warptiling | Adds a warp tile `(WSUBN·WNITER)×(WSUBM·WMITER)`. Each thread does `WNITER·WMITER` sub-tiles of TM×TN | 21779.3 | 93.7% |
| 0 | cuBLAS | | 23249.6 | 100% |

- **Arithmetic intensity drives everything.** Per output element:
  - Kernel 3 → kernel 4: GMEM accesses go from K/16 to K/32, SMEM accesses from 2K to 9K/8.
  - Kernel 5: GMEM K/64, SMEM K/4.
  - Kernel 3 was bound on SMEM instruction issue ("MIO Throttle" stalls, LDS dominating the instruction mix), not on occupancy.
- **Coalescing is a runtime hardware property.** Kernel 2's SASS was unchanged. The compiler can't prove alignment, which is why the `reinterpret_cast<float4*>` in kernel 6 is needed to get 128-bit instructions.
- **Warptiling** (kernel 10) adds a level matching the hardware's warp scheduling. It improves register-cache locality and sets up double buffering. This block→warp→thread hierarchy is the template for [[gemm-tiling-hierarchy]].
- **Tuned parameters are GPU-specific.**
  - A6000: `BM=BN=128 BK=16 TM=TN=8`.
  - A100: `BM=BN=64 BK=16 TM=TN=4` at 12.6 TFLOPS. The A6000 config gets 12 TFLOPS there, and cuBLAS 14.7.
  - The author couldn't explain why specific values win.
- **Thread-block swizzling for L2 gave nothing here,** because the L2 hit rate was already ~80%.

## Numbers worth remembering
- Napkin math at 4092²:
  - Work: 137 GFLOP and at least 268 MB of GMEM traffic.
  - At 30 TFLOPS FP32 and 768 GB/s: ~4.5 ms compute vs ~0.34 ms memory, so the problem is compute-bound.
  - A naive kernel without caching would move ~548 GB.
- cuBLAS's best kernels reach ~245 FLOP/B arithmetic intensity (kernel-3 roofline discussion).
- Kernel 3 occupancy: 37 regs/thread, 1024 threads/block, 8192 B SMEM (+1024 B reserved). That's 1 block/SM, 32 of 48 warps, ~66%.
- **cuBLAS kernel choice on A100 by square size:**

| Size | Kernel | Time |
|---|---|---|
| 128 | `ampere_sgemm_32x32_sliced1x4_nn` | 15.295 µs |
| 256 | `ampere_sgemm_64x32_sliced1x4_nn` + `splitKreduce_kernel` | 12.416 + 6.912 µs |
| 512 | `ampere_sgemm_32x32_sliced1x4_nn` | 41.728 µs |
| 1024–4096 | `ampere_sgemm_128x64_nn` | |

  There were 16 distinct SGEMM kernels across square sizes up to 4096.

## Code patterns
2D blocktiling inner loop (kernel 5/6 shape; paraphrased from the post):
```cuda
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  for (uint i = 0; i < TM; ++i) regM[i] = As[dotIdx * BM + threadRow * TM + i]; // As stored transposed → LDS.128
  for (uint i = 0; i < TN; ++i) regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
  for (uint m = 0; m < TM; ++m)                // outer product: TM*TN FFMAs per TM+TN loads
    for (uint n = 0; n < TN; ++n)
      threadResults[m * TN + n] += regM[m] * regN[n];
}
```
- The `float4` cast `reinterpret_cast<float4*>(&A[...])[0]` asserts 16-byte alignment to the compiler. Without it you get scalar LDG.
- Warp ID is `threadIdx.x / warpSize`. **The post has a typo using `%`.**

## Pitfalls
- The table and the prose disagree for kernel 3: 2980 GFLOP/s in the table vs "~2200 GFLOPS" in the text.
- Tile quantization: grids sized with CEIL_DIV waste work when dimensions aren't tile multiples.
- Bank-conflict-avoidance experiments (kernels 7–8) were slower. Don't assume the profiler's bank-conflict counter is your bottleneck → [[smem-bank-conflicts]].
- Hand kernels fall off against cuBLAS at small sizes, because cuBLAS switches to different tiles and to split-K.

## Relevance to the serving track
- **Transfers to small m (inference):**
  - Coalesced 128-bit global loads of W. For m ≤ 256, W traffic is essentially all the traffic, so this is the single most important item from the post.
  - Register tiling across N, so each loaded x value is reused across several W rows.
  - The observation that cuBLAS goes split-K at small sizes → [[split-k-and-stream-k]].
  - Autotuning per shape (the user's `tuned_*.json` approach).
- **Does not transfer (inference):**
  - FFMA-based 2D blocktiling at 8×8 per thread aims to make a large GEMM compute-bound. At m=1–8 there are no 8 rows of A to tile over.
  - BF16 on Hopper should use tensor cores (`mma.sync` or wgmma) for m ≥ 8–16. A CUDA-core dot-product GEMV is competitive only at m=1–4 or so (see [[gemv-and-skinny-gemm]]).
  - L2 block swizzling matters even less: a skinny GEMM reads each W byte once.
- The kernel-3 lesson ("profile the stall reason: MIO throttle ≠ memory-bound") applies directly when a skinny kernel isn't reaching HBM bandwidth.

## Related
[[gemm-tiling-hierarchy]] · [[register-tiling]] · [[alexarmbr-tensor-core-hgemm]] · [[hidden-treasures-3-gemm-code]] · [[split-k-and-stream-k]] · [[gemv-and-skinny-gemm]] · [[smem-bank-conflicts]] · [[roofline-and-arithmetic-intensity]] · [[hidden-treasures-llms]]
