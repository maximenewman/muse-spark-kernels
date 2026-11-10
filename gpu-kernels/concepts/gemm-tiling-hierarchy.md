---
title: "GEMM tiling hierarchy: block / warp / thread / mma tiles, and how to pick them for small M"
type: concept
tags: [gpu, gemm, tiling, tensor-cores, occupancy, small-m, skinny-gemm]
source_url:
  - https://siboehm.com/articles/22/CUDA-MMM
  - https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
  - https://github.com/Bruce-Lee-LY/cuda_hgemm
  - raw/hidden-treasures-llms/3-GEMM/
---
# GEMM tiling hierarchy

**TL;DR**
- Four nested tiles map onto the memory hierarchy:
  - **Block tile** (BM×BN×BK): GMEM → SMEM, one CTA, one SM.
  - **Warp tile** (WM×WN×WK): SMEM → registers, one warp.
  - **Thread tile** (TM×TN): CUDA-core path only.
  - **MMA tile**: m16n8k16 (bf16) per `mma.sync`.
- **Big GEMM** picks tiles to maximize **arithmetic intensity** (block: `BM·BN/(BM+BN)` FLOP/B; alexarmbr). Typical sizes: 128–256 × 128–256 block, 64×64 warp.
- **Small M (1–256)** flips the objective. Intensity is capped at ≈ m FLOP per weight byte no matter how you tile, so pick tiles to **(1) fill every SM, (2) keep enough W bytes in flight, (3) avoid wasted mma rows and registers** (inference).
- Practical small-m recipe (inference):
  - BM = m rounded up to 16 (or swap AB so tokens are the mma-N=8 side).
  - BN = 64–128 W rows.
  - BK = 64–256.
  - Split-K so that `ceil(N/BN) × splitK ≳ 1–2 × #SMs`.

## Key ideas
### The levels (sourced)
| Level | Moves data | Example sizes from sources | What it buys |
|---|---|---|---|
| Block | GMEM→SMEM, shared by all warps in the CTA | siboehm A6000 tuned 128×128×16; alexarmbr final 256×256×32; Bruce-Lee 256×128×32 | Reuse of A across BN and of B across BM. GMEM traffic per output ∝ 1/BN + 1/BM |
| Warp | SMEM→registers | alexarmbr kernel 1: 64×64 (code kernel 5: 128×64); Bruce-Lee 64×64 | SMEM reads per mma. Warp-level intensity (alexarmbr: 64×64 → 32 FLOP/B vs SMEM balance 17.7) |
| Thread (CUDA cores) | Registers | siboehm TM=TN=8 (64 accumulators/thread) | FFMA per LDS: TM·TN FMAs per TM+TN loads |
| MMA | Registers → tensor core | m16n8k8 (Turing f16), m16n8k16 (Ampere+ f16/bf16) | Fixed by hardware. Warp tile = an integer grid of these |

- siboehm's warptiling (kernel 10) inserts the warp level between the block and thread levels. It gives +10% and makes double buffering possible.
- Constraints that bind in practice:
  - **SMEM per stage** = (BM+BN)·BK·bytes × stages.
  - **Registers:** accumulators = WM·WN/32 floats per thread for f32. 128×64 warp tile → 256 f32 regs, too many. alexarmbr's 128×64 works only because the accumulators are f16x2: `acc[8][8][2]` u32 = 128 regs.
  - **Divisibility:** the code in [[hidden-treasures-3-gemm-code]] asserts `M % BM == 0`.
- **Tile size vs problem size:** in [[hidden-treasures-3-gemm-code]], 256×256 tiles are *slower* than 256×128 at 1024³. Inference: 16 CTAs on 40 SMs. cuBLAS changes tile shape with size and uses split-K at 256³ ([[siboehm-sgemm-worklog]]).

### Choosing tiles when M = 1…256 (inference unless noted)
For `out[m,N] = x[m,K] @ W[N,K]ᵀ`, and assuming W is read once and m ≤ 256, total bytes ≈ 2·N·K. The kernel is HBM-bound. Using H100 SXM ≈ 3.35 TB/s and H200 ≈ 4.8 TB/s (NVIDIA spec sheets, not these sources), the floor is time ≈ 2NK / BW.

1. **BM (token side).** m ≤ 16 → one mma row-tile (15/16 waste at m=1, irrelevant to time). Better: **swap AB**, putting tokens on mma-N (8 per mma). m=8 is then exact, and m=16/32 is 2/4 n-tiles. For m=128–256, BM=64–128 tokens with WM=16–32 per warp. Watch registers: f32 acc = (WM·WN)/32 per thread.
2. **BN (W rows per CTA).** Choose it so the grid fills the GPU: `CTAs = ceil(N/BN) × splitK`, H100/H200 have 132 SMs (spec). Example: qkv_gate N=2304, BN=128 gives 18 CTAs, so split-K ≈ 8 to reach ~144 CTAs. pooler N=4096, BN=128 gives 32 CTAs, so split-K 4–8. The shapes are in [[muse-spark-open-model]].
3. **BK and stages.** Make each CTA's in-flight W bytes large: BN × BK × 2 B × stages, e.g. 128×128×2×3 ≈ 96 KB. But keep ≥ 2–4 K-iterations per CTA after split-K, or the pipeline never reaches steady state.
4. **Warp tile.** Warps should split **N** (W rows) and/or **K** (intra-CTA split-K), not M. With m=8 there's nothing to split in M. An intra-CTA K split then needs an SMEM reduction of f32 partials across warps.
5. **MMA tile.** bf16 m16n8k16 with f32 accumulate. Swap-AB mapping: A = W (16 rows × 16 k, `ldmatrix.x4`), B = x (16 k × 8 tokens, `ldmatrix.x2`) → [[mma-ptx]].
6. **m=1 (maybe up to ~4):** tensor-core tiling can be skipped entirely. A warp-per-row (or several rows) dot-product GEMV with 128-bit loads and a shuffle reduction is simpler and equally HBM-bound → [[gemv-and-skinny-gemm]].

## Numbers worth remembering
- Block intensity `BM·BN/(BM+BN)` FLOP/B (alexarmbr): 256×128 → 85.3, 256×256 → 128.0. T4 balance vs GMEM is 224 FLOP/B for tensor cores and ~33 for FFMA.
- A6000 SGEMM autotune winner `BM=BN=128 BK=16 TM=TN=8`. A100: `BM=BN=64 BK=16 TM=TN=4` (siboehm).
- Bruce-Lee: 256×128 block, 64×64 warp, 8 warps, 32-wide K chunk per stage.

## Code patterns
Loop nest (alexarmbr structure; comments mine):
```cuda
for (block_k ...) {                         // GMEM→SMEM (cp.async/TMA on sm_80+/90)
  load A_tile[BM][BK], B_tile[BK][BN]; sync;
  for (warp_k ...) {                        // SMEM→regs via ldmatrix
    for (mma_m) for (mma_k) ldmatrix A_frag[mma_m][mma_k];
    for (mma_k) for (mma_n) ldmatrix B_frag[mma_k][mma_n];
    for (mma_k) for (mma_n) for (mma_m)     // outer product of fragments
      mma(acc[mma_m][mma_n], A_frag[mma_m][mma_k], B_frag[mma_k][mma_n]);
  }
  sync;
}
```
Grid sizing for a skinny GEMM (inference):
```python
sms = 132
ctas_n = cdiv(N, BN)
split_k = max(1, min(cdiv(target_waves * sms, ctas_n), K // (BK * min_k_iters)))
```

## Pitfalls
- Copying big-GEMM tile configs (128×256 etc.) to m ≤ 256 leaves most SMs idle, and wastes registers on padded M rows (inference; consistent with the 1024³ regression above).
- Wave quantization: CTAs = 1.1 × SMs costs nearly 2 waves. Pick split-K and BN to land just under an integer multiple of the SM count, or use stream-K → [[split-k-and-stream-k]].
- Thread tiles for CUDA-core GEMMs (siboehm) don't map to tensor-core kernels. There the mma fragment layout fixes per-thread ownership.

## Relevance to the serving track
- **Sourced facts that transfer:** the level structure, the SMEM/register budgets, the `ldmatrix`/mma fragment shapes, and "cuBLAS switches to split-K for small problems".
- **Inference:** for `muse_bf16_proj`, the decision variables are swap-AB (yes for m ≤ 64?), BN, split-K, BK, stages and warps/CTA. Tune them **per (N, K, m) under CUDA-graph replay with cold L2**, as the user's `tuned_*.json` already does → [[benchmarking-kernels]].

## Related
[[mma-ptx]] · [[cp-async]] · [[register-tiling]] · [[split-k-and-stream-k]] · [[gemv-and-skinny-gemm]] · [[siboehm-sgemm-worklog]] · [[alexarmbr-tensor-core-hgemm]] · [[bruce-lee-mma-ptx]] · [[hidden-treasures-3-gemm-code]] · [[roofline-and-arithmetic-intensity]] · [[persistent-kernels-and-tile-scheduling]] · [[muse-spark-open-model]]
