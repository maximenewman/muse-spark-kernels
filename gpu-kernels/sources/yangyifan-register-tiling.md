---
title: "Yifan Yang – How to Make a Compute-bound Problem Actually Compute-bound"
type: source
tags: [gpu, gemm, gemv, tiling, register-tiling, roofline, multicast, bandwidth-amplification]
source_url: https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html
---
# Yifan Yang – How to Make a Compute-bound Problem Actually Compute-bound

Fetched in full 2026-11-08. Analytical, no profiling. Uses FP32 CUDA-core GEMM on A100 as the running example plus a GEMV bonus section.

**TL;DR**
- Think in **throughput matching**, not latency hiding: `latency = static_latency + tile_size/BW`, so throughput is first-class.
- Model the hierarchy DRAM→smem→RF→cores with finite capacity and BW per level. Each stage must deliver the cores' input demand; the gap is closed by **bandwidth amplification** = data reuse.
- Reuse factor per input element for an output-stationary tile BM×BN: A reused BN times, B reused BM times; average `2·BM·BN/(BM+BN)`. Required reuse at a level = (demand BW)/(supply BW of that level).
- Two optimizations (smem tiling + RF tiling) got 88% of cuBLAS on V100.
- Multicast (cluster TMA multicast, L2 request coalescer, smem broadcast) is a second amplification mechanism; rasterization order controls how much sharing exists.
- **GEMV**: no RF reuse (reuse = 1); it only reaches the memory roofline because the **L2 Request Coalescer (LRC)** de-duplicates every SM's reads of the shared vector.

## Key ideas
- Simple roofline (infinite on-chip storage + infinite on-chip BW) says 4096³ GEMM is compute-bound but can't tell you why naive code gets 16% of cuBLAS.
- Per-SM A100 model: DRAM→SM 13.3 B/cycle (2039 GB/s / (108 SMs × 1.41 GHz)); smem→RF 128 B/cycle; cores need 512 B/cycle (64 FP32 FMA/cycle × 2 operands × 4 B).
- Required amplification: DRAM→smem 512/13.3 ≈ 39 → BM=BN≥39 → pick 64. smem→RF 512/128 = 4 → RF tile 4×4 per thread.
- **Output-stationary (OS) dataflow** everywhere ⇒ output BW negligible, only input BW matters. Used because inter-SM communication via gmem/L2 is expensive (DSMEM in Hopper might change that).
- **Local vs global reuse**: with L1=16 (smem-level local) and L2=4 (RF-level), global smem reuse = 64; think in global reuse per stage to avoid confusion. Loop nest in post shows the three levels.
- **Multicast**: 4 SMs with 32×32 tiles alone need 19.2K cycles of DRAM time vs 16K compute (BW-bound); with a 2×2 cluster sharing 2 A + 2 B tiles → 9.6K, now compute-bound. Equivalent views: multicast doubles DRAM BW, or makes the effective cluster tile 64×64.
- **Rasterization** (4×4 tiles, 4 SMs, 32×32): row-wise wave multicast factor 256/160 = 1.6; 2×2 square wave 256/128 = 2; adversarial = 1.
- HW multicast mechanisms: threadblock cluster (user-specified); **LRC** (HW opportunistically merges duplicate L2 requests); smem broadcast within a warp (LDS.64/128 overlapping addresses).
- Other amplification: more HBM, compression, sparsity, **fusion** (batched GEMM sharing A).
- Practice factors: power throttling (bigger tiles reduce data-movement energy), static memory latency growing per HBM gen (need MLP → multi-stage pipelines, `num_stages`), DRAM bank camping, smem bank conflicts.

## Numbers worth remembering
| Kernel (V100, FP32 CUDA core, 4096³) | TFLOPS | % cuBLAS |
|---|---|---|
| Naive | 1.9 | 16% |
| smem tiling | 3.7 | 31% |
| RF tiling | 10.7 | 88% |
| cuBLAS | 12.1 | 100% |
- A100 80GB SXM: 2039 GB/s, 108 SMs, 1.41 GHz → 13.3 B/cycle/SM; 64 FP32 FMA/cycle/SM; machine balance 4.8 FMA/B.
- 4096³ GEMM intensity: 64 G FMA / 48 MB = 1365 FMA/B.
- FP32 GEMV (BM=BN=1, direct DRAM→RF): realistic model 13.3/8 = 1.64 FMA/cycle/SM vs roofline ¼ FMA/B × 13.3 = 3.33 FMA/cycle/SM; LRC's ×2 on the shared vector closes the gap.

## Code patterns
```python
# three-level OS loop nest from the post (L1 = smem-level local reuse, L2 = RF-level reuse)
for m0 in range(M // L1 // L2):
  for n0 in range(N // L1 // L2):                       # DRAM
    sA = gA[m0*L1*L2:(m0+1)*L1*L2, :]; sB = gB[:, n0*L1*L2:(n0+1)*L1*L2]
    for m1 in range(L1):
      for n1 in range(L1):                              # smem
        rA = sA[m1*L2:(m1+1)*L2, :]; rB = sB[:, n1*L2:(n1+1)*L2]
        for m2 in range(L2):
          for n2 in range(L2):                          # RF
            for k in range(K):
              C[m0*L1*L2+m1*L2+m2, n0*L1*L2+n1*L2+n2] += rA[m2,k]*rB[k,n2]
```

## Pitfalls
- Optimizing for latency hiding first hides the real constraint (stage throughput).
- A tile size that just meets the minimum reuse is not enough when power-throttled; bigger is better there.
- Bad rasterization destroys multicast/L2 sharing (though that adversarial pattern is what smem swizzling wants — see [[swizzling]]).

## Relevance to the serving track
- `x[m,k] @ W.T` with m ≤ 256: W reuse factor = m (each W element is used by m rows); x reuse = n-tile width. At m=1 this is the post's GEMV: zero RF reuse on W, so the kernel is pure **W streaming** and the only lever is achieving peak DRAM BW (MLP, enough bytes in flight, no wasted bytes). (inference)
- x is shared by every CTA: per the GEMV section the L2 request coalescer de-duplicates it; don't waste smem/TMA effort amplifying x at small m. (inference)
- The "required amplification = demand/supply" recipe tells you when tensor cores are even needed: with BF16 and m=1..32 the FMA demand per byte of W is tiny, so a CUDA-core or small-MMA kernel can be enough; at m=128/256 W reuse grows and tensor cores matter. (inference)
- The latency formula `static_latency + tile/BW` is the cold-L2 story: with few µs total, static DRAM latency is a large fraction, so issue as many independent loads (TMA boxes / L2 prefetches) as early as possible. (inference)

## Further reading
See [[yangyifan-cute-tma]] "Further reading". Author's references: Simon Boehm's CUDA MMM blog, Colfax persistent kernels / stream-K, GTC25 S73619, Axel Feldmann's smem microbenchmarks, Horace He on power throttling.

## Related
[[register-tiling]] [[tma]] [[smem-bank-conflicts]] [[swizzling]] [[tensor-core-generations]]
