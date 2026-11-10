---
title: Register Tiling and Bandwidth Amplification
type: concept
tags: [gpu, gemm, gemv, tiling, register-tiling, roofline, data-reuse, memory-bound]
source_url:
  - https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html
---
# Register Tiling and Bandwidth Amplification

**TL;DR**
- Every level of the hierarchy (DRAM→smem→RF→ALU/TC) must supply the compute units' operand demand; where a level's BW is lower, **data reuse (tiling)** amplifies it. Required reuse at a level = demand BW / that level's supply BW.
- Output-stationary tile BM×BN reuses each A element BN times and each B element BM times; average reuse `2·BM·BN/(BM+BN)`.
- **RF tiling** (each thread owns a small outer-product tile, e.g. 4×4) amplifies smem→RF BW; **smem tiling** (CTA tile, e.g. 64×64) amplifies DRAM→smem BW. Together: 88% of cuBLAS for FP32 SGEMM on V100.
- Multicast (TMA cluster multicast, L2 request coalescer, smem broadcast) amplifies BW without larger per-SM tiles.
- GEMV/skinny GEMM has reuse ≈ 1 on the matrix: no tiling can help; it's purely a bandwidth/latency problem.

## Key ideas
- **Throughput first**: `latency = static_latency + tile_size/BW`; latency hiding is a consequence of throughput matching.
- **Machine model** (A100 per SM, FP32 CUDA cores): DRAM 13.3 B/cycle, smem→RF 128 B/cycle, ALU demand 512 B/cycle ⇒ need ×39 at DRAM→smem (BM=BN≥39 → 64) and ×4 at smem→RF (4×4 RF tile).
- **Global vs local reuse**: compute the reuse requirement per stage relative to the final consumer demand (global), not relative to the next level's BW. Local reuse 16 at smem × 4 at RF = global 64 = smem tile size.
- **OS dataflow** at all levels makes output BW negligible; if you break OS (split-K, stream-K), output traffic must be modelled too.
- **Multicast**: needs multiple consumers requesting the same data; 2×2 cluster of 32×32 tiles behaves like a 64×64 tile for DRAM traffic. Rasterization order determines the sharing factor (row-wise 1.6× vs 2×2 square 2× in the 4-SM example).
- HW multicast: cluster TMA multicast (explicit), **L2 Request Coalescer** (implicit, merges duplicate requests), smem broadcast to a warp (LDS.64/128 overlapping addresses).
- Real-world extras: power throttling (bigger tiles save data-movement energy), growing static HBM latency (need MLP → multistage pipelines), DRAM bank camping, smem bank conflicts.
- **GEMV case**: reuse on the matrix = 1; per-SM FP32 GEMV naively reaches 1.64 FMA/cycle vs roofline 3.33 — the L2 Request Coalescer serving the shared vector to all SMs is what makes the roofline achievable (most DRAM BW then goes to the matrix).

## Numbers worth remembering
- V100 FP32 4096³: naive 1.9, smem-tiled 3.7, +RF-tiled 10.7, cuBLAS 12.1 TFLOPS.
- A100 80GB SXM: 2039 GB/s, 108 SMs, 1.41 GHz, 13.3 B/cycle/SM, 64 FP32 FMA/cycle/SM, balance 4.8 FMA/B.

## Code patterns
```cuda
// RF-tiled inner loop (canonical form of the post's RF level; TM=TN=4 gives reuse 4)
float acc[TM][TN] = {};
for (int kk = 0; kk < BK; ++kk) {
  float a[TM], b[TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i) a[i] = sA[kk][ty*TM + i];   // TM smem loads
  #pragma unroll
  for (int j = 0; j < TN; ++j) b[j] = sB[kk][tx*TN + j];   // TN smem loads
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j]; // TM*TN FMAs per (TM+TN) loads
}
```
(Illustrative; the post presents the loop nest in Python, see [[yangyifan-register-tiling]].)

## Pitfalls
- Optimizing latency hiding (occupancy tricks) before achieving per-stage throughput match.
- Minimum-reuse tiles are not enough under power throttling; and large RF tiles raise register pressure (occupancy) — balance. (second half: inference)
- Rasterization that maximizes multicast is the adversarial pattern for smem bank conflicts; handle conflicts with [[swizzling]] instead.

## Relevance to the serving track
- `x[m,k] @ W[n,k]ᵀ`, BF16: W reuse = m (one W element feeds m outputs), x reuse = n-tile width. For m=1..32, W bytes dominate traffic and reuse is tiny ⇒ memory-bound; the optimal kernel is a **W-streaming** kernel whose job is to reach peak HBM BW with minimal fixed overhead. (inference)
- Register tiling still matters inside the CTA: hold the whole m×BK slice of x in registers/smem and reuse it across many W rows (x reuse = rows per CTA), so the only DRAM stream is W. (inference)
- x is read by every CTA; per the GEMV analysis the L2 coalescer dedups it, so don't spend smem/cluster machinery on x at small m. (inference)
- At m=128/256 reuse on W reaches 128–256, approaching the compute-bound regime where tensor cores ([[wgmma]]) and proper smem tiling matter again. (inference)
- Cold L2 + µs-scale kernels: the static-latency term dominates; maximise bytes in flight from t=0 (deep TMA ring, L2 prefetch, start before `griddepcontrol.wait` via [[pdl]]). (inference)
- Split-K becomes attractive when n/BN CTAs < SM count (e.g. small-n layers like router), but breaks OS dataflow → output traffic/reduction cost must be counted. (inference)

## Related
[[yangyifan-register-tiling]] [[tma]] [[pdl]] [[cute-layouts]] [[smem-bank-conflicts]] [[swizzling]] [[wgmma]] [[mma-ptx]] [[tensor-core-generations]]
