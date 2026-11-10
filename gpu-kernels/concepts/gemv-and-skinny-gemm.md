---
title: "GEMV and skinny GEMM (M = 1…256): weight streaming, split-K, mma with padded M vs CUDA-core dots"
type: concept
tags: [gpu, gemv, skinny-gemm, small-m, decode, memory-bound, split-k, swap-ab, bf16, muse]
source_url:
  - https://siboehm.com/articles/22/CUDA-MMM
  - https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
  - https://github.com/Bruce-Lee-LY/cuda_hgemm
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
---
# GEMV and skinny GEMM (M = 1…256)

**TL;DR**
- For `out[m,N] = x[m,K] @ W[N,K]ᵀ` with m ≤ 256, **W dominates the bytes** (2·N·K for bf16) and is read once. FLOPs are 2·m·N·K, so **intensity ≈ m FLOP/B**, far below the H100/H200 BF16 ridge (inference from the roofline idea in [[alexarmbr-tensor-core-hgemm]] / [[siboehm-sgemm-worklog]]). **The kernel is HBM-bound, and the floor is 2NK / BW.**
- **The job is to stream W at full HBM bandwidth:**
  - enough CTAs to cover all SMs (→ [[split-k-and-stream-k]])
  - enough bytes in flight per SM (→ [[cp-async]], [[tma]], unrolled 128-bit LDG)
  - coalesced 16 B loads along K
  - a cheap reduction and epilogue
  - nothing else on the critical path (launch, memset, extra reduce kernels → [[cuda-graphs]], [[pdl]])
- **Tensor cores vs CUDA cores (inference):**
  - m=1: a CUDA-core dot product (warp per row, shuffle reduce) is fine.
  - m ≥ ~8: `mma.sync m16n8k16` with **swap-AB** (W rows → mma-M=16, tokens → mma-N=8) avoids per-element FMA/convert overhead and wastes nothing at m=8.
  - Between m=2 and m≈4–8, measure both.
- Big-GEMM tricks that **don't** help: huge block tiles, L2 block swizzle, deep M reuse, FP16 accumulate. Tricks that **do**: split-K, swap-AB, multistage W prefetch, fused/self-resetting reductions, per-shape tuning under graph replay.

## Key ideas
### Why memory-bound (inference, using spec-sheet bandwidth)
- Bytes ≈ 2NK (W) + 2mK (x) + 2mN (out) + split-K partials (4·S·m·N, written and read). FLOPs = 2mNK.
- H100 SXM ≈ 3.35 TB/s and H200 ≈ 4.8 TB/s HBM (NVIDIA spec sheets; not from these sources). The dense-BF16 tensor ridge is hundreds of FLOP/B, so even m=256 sits near or below the ridge, and m ≤ 32 is deep in memory-bound territory. `mma.sync` on Hopper runs below the wgmma peak, so check m=128–256 against the compute side too → [[wgmma]].
- `muse_bf16_proj` floors (W bytes only; H200 at 4.8 TB/s, shapes from [[muse-spark-open-model]]):

| Proj | W shape [N,K] | W bytes | floor @4.8 TB/s |
|---|---|---|---|
| qkv_gate | [2304, 8192] bf16 | 37.7 MB | ~7.9 µs |
| o | [8192, 1024] bf16 | 16.8 MB | ~3.5 µs |
| pooler | [4096, 8192] bf16 | 67.1 MB | ~14.0 µs |
| unpooler (m≤8 slice) | [1024, 4096] bf16 | 8.4 MB | ~1.75 µs |
| unpooler (full) | [8192, 4096] bf16 | 67.1 MB | ~14.0 µs |
| router | [256, 8192] fp32 | 8.4 MB | ~1.75 µs |

These are consistent with the challenge's stated 2–17 µs floors (inference). With **cold L2** timing, every W byte really comes from HBM.

### Design space (sourced building blocks, inference for the assembly)
1. **Operand mapping** ([[mma-ptx]]):
   - (a) x as A (pad m to 16), W as B via `ldmatrix.x2` (no trans, W is K-contiguous).
   - (b) **swap-AB**: W as A via `ldmatrix.x4` (16 rows × 16 k), x as B (`ldmatrix.x2`, 8 tokens). C comes out as out[n][token]; transpose on store.
   - The user's kernel-track Triton kernel uses swap-AB ([[findings-log]]).
2. **Grid fill:** `ceil(N/BN) × S ≥ ~#SMs` (132 on H100/H200). Small N needs larger S, and S shrinks as m grows → [[split-k-and-stream-k]].
3. **W delivery**, three options:
   - (i) `cp.async.cg` 16 B multistage ring into swizzled SMEM, then `ldmatrix` ([[bruce-lee-mma-ptx]], [[cp-async]]).
   - (ii) TMA bulk tiles with mbarrier on sm_90 ([[tma]]).
   - (iii) **direct LDG.128 into registers in fragment order**, using a consistent k-permutation on W and x. No SMEM, no ldmatrix, no bank conflicts.
     - **Measured: this loses.** The original reasoning (inference) was that W is used exactly once, so SMEM staging buys no reuse, only latency hiding.
     - **That latency hiding is the whole point.** Register-held in-flight bytes cap occupancy and stall on `long_scoreboard`, while (i)/(ii) hold them in SMEM through async copies.
     - Profiled on H100: [[register-direct-weight-streaming]], [[findings-log]] 2026-11-09.
4. **x delivery:** x[m, K-slice] is small (m·K/S·2 B, e.g. 8×2048×2 = 32 KB). Load it once per CTA (or per stage) into SMEM and reuse it for all BN W rows; or keep x fragments in registers when m ≤ 8.
5. **Reduction and epilogue:** f32 accumulate. The split-K reduce should be the last CTA with a self-resetting ticket (no memset node), then a bf16 store. For the FP32 router, use a bf16 hi+lo split with 3 mma products into the same f32 accumulator, or plain f32 CUDA-core FMA at m=1.
6. **CUDA-core GEMV (m=1…~4):**
   - Each warp owns 1–4 W rows. Each lane loads 16 B (8 bf16) per step and converts to f32.
   - FMA against x held in SMEM/registers, then `__shfl_xor_sync` reduce.
   - No tensor-core fragment constraints. W rows can be any count per warp, and K is split across lanes and unrolled for in-flight loads.
   - The FMA rate (~1 per W element per token) stays well under CUDA-core FP32 throughput at m=1, but conversion and FMA instruction count grow ∝ m. Hence the switch to mma for larger m (inference).

### What transfers from the big-GEMM sources
| Technique (source) | Small-m verdict (inference) |
|---|---|
| Coalesced, 128-bit vectorized loads (siboehm k2/k6, alexarmbr k2) | **Essential.** This is the whole game |
| SMEM tiling for reuse (siboehm k3, alexarmbr k1) | x reuse yes; W has no reuse. Stage W only for latency/layout |
| 2D register tiling / big warp tiles (siboehm k5/k10, alexarmbr k5) | Mostly no. Tile N/K, not M |
| Swizzle/padding for `ldmatrix` (alexarmbr k3, Bruce-Lee permuted) | Needed only if W goes through SMEM → [[swizzling]] |
| Register prefetch / cp.async multistage (alexarmbr k4/k6, Bruce-Lee stage3/4) | **Yes.** Bytes in flight decide achieved BW |
| XOR-stepped addresses, low index math (alexarmbr k5) | Yes. Integer overhead per W byte matters at full BW |
| Large block tiles 128–256 (all) | **No.** The 1024³ regression in [[hidden-treasures-3-gemm-code]] shows big tiles starving SMs |
| L2 thread-block swizzle (siboehm k10, Bruce-Lee) | No. W is read once |
| Split-K (cuBLAS at 256³, siboehm) | **Yes**, primary |
| FP16 accumulate (alexarmbr, Bruce-Lee) | No. BF16 mma requires f32 accumulate |

## Measured on H100 (2026-11-08, [[findings-log]])
- **Tuned Triton beat a hand-written register-direct CUDA kernel** at every BF16 shape for m ≤ 32 ([[register-direct-weight-streaming]]). Start with Triton.
- **ncu explains it:** Triton's m ≥ 8 kernel runs one warp per scheduler with a 5-stage `cp.async` ring (80–130 KB SMEM/CTA), and stalls ~2.5 cycles per issue against 20+ for register-direct.
- A hand-written `cp.async` ring got within 5–10% of Triton, **but only after x was staged through the ring as well.**
- Best H100 configs (µs, cold L2, graph replay):

| shape | config | time | floor |
|---|---|---|---|
| m=1 qkv_gate | CUDA-core GEMV, BN=4 rows, BK=1024, no split, 576 CTAs | 12.9–13.2 | 11.3 |
| m=1 pooler | GEMV, BN=8, BK=512, no split | 22.7–23.7 | 20.0 |
| m=8–32 | swap-AB `tl.dot`, BN=32–64, BM=16/32, BK=128–256, split 1–2, 4 warps, 5 stages | — | — |
| router (FP32 W) | split 8–16 | — | — |

- **m = 128/256 BF16: cuBLAS wins** over both Triton orientations. Use a TMA + [[wgmma]] warp-specialized kernel or keep cuBLAS.

## Numbers worth remembering
- Arithmetic intensity of skinny GEMM ≈ m FLOP per W byte (bf16), or m/2 for an fp32 W (inference, from the definitions).
- cuBLAS A100 256³ SGEMM = split-K GEMM 12.416 µs + reduce 6.912 µs (siboehm). A separate reduce kernel is expensive at µs scale.
- H100/H200: 132 SMs (spec). Decode shapes: m ∈ {1, 16, 32} decode, {8, 128, 256} verify ([[muse-spark-open-model]]).

## Code patterns
CUDA-core GEMV row-dot (inference; the m=1 path):
```cuda
// one warp computes out[0, n] for one W row n; W row-major [N,K] bf16, x [K] bf16 in smem
const uint4* w = reinterpret_cast<const uint4*>(W + (size_t)n * K);   // 16 B = 8 bf16 per load
float acc = 0.f;
#pragma unroll 4
for (int i = lane; i < K / 8; i += 32) {          // coalesced: 32 lanes x 16 B = 512 B per step
  uint4 wv = w[i];                                 // LDG.128 (consider ld.global.nc / L1::no_allocate)
  uint4 xv = reinterpret_cast<const uint4*>(xs)[i];
  const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
  const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    float2 a = __bfloat1622float2(wp[j]), b = __bfloat1622float2(xp[j]);
    acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
  }
}
for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
if (lane == 0) out[n] = __float2bfloat16(acc);
```
Swap-AB mma tile (inference; operands as in [[mma-ptx]]):
```cuda
// A = 16 W rows x 16 k   (ldmatrix.x4 from smem W tile, or direct LDG in permuted-k order)
// B = 16 k x 8 tokens    (ldmatrix.x2 from smem x tile [token][k])
// C = 16 W rows x 8 tokens f32: thread (g,t) holds out[token 2t, 2t+1][n0+g], [..][n0+g+8]
mma_bf16_16816(c, a, b);   // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

## Pitfalls
- **Wave quantization and tiny grids:** N/BN alone is often < 132 → split-K, or several W-row groups per CTA with a persistent loop.
- **Partial traffic grows with m·S.** At m=256 large S can double the bytes moved ([[split-k-and-stream-k]]).
- **Zeroing workspaces** inside graphs costs a node. Prefer self-resetting tickets.
- **Timing:** eager-launch timing misleads at µs scale. Configs that win eagerly can lose under graphs, so tune under graph replay with cold L2 ([[findings-log]], [[benchmarking-kernels]]).
- Using CUDA-core GEMV at m=8+ multiplies the FMA and convert instruction count by m, and you become issue-bound before HBM-bound (inference).
- Padding m to 16 in the A slot doubles the A-fragment register count compared with swap-AB at m ≤ 8 (inference).

## Relevance to the serving track
This note *is* the serving-track view. Summary of levers for `muse_bf16_proj`, ranked by expected payoff (inference):
1. Per-shape split-K S + BN to fill 132 SMs.
2. Multistage W prefetch (cp.async or TMA) or a register-direct W stream with deep unroll.
3. Swap-AB `mma.sync` m16n8k16 bf16 for m ≥ 8, CUDA-core GEMV at m=1.
4. Fused last-CTA reduction, no memset.
5. PDL so the next kernel's prologue overlaps this tail ([[pdl]]).
6. For m = 128–256, check whether [[wgmma]] beats `mma.sync`.

## Related
[[split-k-and-stream-k]] · [[gemm-tiling-hierarchy]] · [[mma-ptx]] · [[cp-async]] · [[tma]] · [[wgmma]] · [[pdl]] · [[cuda-graphs]] · [[swizzling]] · [[memory-bandwidth-and-latency]] · [[roofline-and-arithmetic-intensity]] · [[weight-only-quantization]] · [[muse-spark-open-model]] · [[serving-track-playbook]] · [[findings-log]] · [[siboehm-sgemm-worklog]] · [[alexarmbr-tensor-core-hgemm]] · [[bruce-lee-mma-ptx]] · [[hidden-treasures-3-gemm-code]]
