---
title: "alexarmbr: How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores"
type: source
tags: [gpu, gemm, hgemm, tensor-cores, mma-ptx, ldmatrix, swizzling, turing, t4]
source_url:
  - https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
  - https://github.com/alexarmbr/matmul-playground
---
# alexarmbr: Tensor-core HGEMM from scratch

**TL;DR**
- FP16 HGEMM `D = αAB + βC` on a **Tesla T4 (Turing, sm_75)** using raw PTX: `ldmatrix` plus `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16`, no WMMA. Six kernels reach **~96% of cuBLAS at 8192³**.
- Progression:
  1. Hierarchical tiling (~8% of cuBLAS).
  2. Vectorized/unrolled gmem→smem copy (~3×).
  3. **SMEM swizzle** (~2×, ~50% of cuBLAS).
  4. Register prefetch of the next block tile (~70%).
  5. Retuned tiles and XOR-only index math (86.7%).
  6. Double-buffered SMEM (~96%).
- Core lesson: **tensor cores need ~6.6× the arithmetic intensity of FFMA**. Swapping FFMA for HMMA in a CUDA-core-optimized kernel isn't enough. After that, **index math competes with HMMA for issue slots**.
- The author **chose PTX over WMMA because WMMA hides the smem→register loads** (see [[bruce-lee-mma-ptx]], [[mma-ptx]]).
- The code lives locally at `raw/hidden-treasures-llms/3-GEMM/`. The per-kernel diff walkthrough is in [[hidden-treasures-3-gemm-code]].

## Key ideas
- **Loop nest:** GPU → thread block (one SM) → warp → `mma` tile. Each `block_k` step copies a block tile to SMEM. Each `warp_k` step `ldmatrix`es warp tiles into registers. The innermost loop is an outer product of mma tiles. See [[gemm-tiling-hierarchy]].
- **Block-level intensity** = `(BM·BN)/(BM+BN)` FLOP/B. BK cancels out.
  - BM=256, BN=128 gives 85.3, too close to the 87.24 balance point.
  - BM=BN=256 gives 128.0.
- **Warp-level intensity** for WM=WN=64 is 32 FLOP/B, against an SMEM balance of 17.7, so SMEM shouldn't dominate.
- **`ldmatrix`** loads 8×8 b16 tiles from SMEM straight into the per-thread fragment layout that `mma` expects. It can **transpose on the fly** (`.trans`), which suits the column-major B operand.
- **Bank conflicts:** with a row-major tile whose row stride is a multiple of 128 B, the 8 rows an `ldmatrix` 8×8 reads all land in the same 4 banks. That's an 8-way conflict (measured wavefront ratio ~8). Two fixes:
  - **Padding** each row by 16 B costs 6144 B of SMEM in kernel 2.
  - An **XOR swizzle** costs ~4 lines.
  - Toy swizzle: `f(i) = i ^ ((i & 0b1100) >> 2)`. The real one XORs the bits that select the 16-byte column chunk with the row-within-8 bits.
  - The author credits CUTLASS's `Swizzle<bits, base, shift>` and wrote `scripts/shmem_layout_viz.py`. → [[swizzling]], [[smem-bank-conflicts]]
- **Bank conflicts are hidden in Nsight Compute.** They show up as L1/TEX throughput and **MIO throttle** stalls. Read the shared wavefront counts (actual vs ideal).
- **Turing has no `cp.async`.** Kernel 4 fakes it: issue the `LDG.128` for tile k+1 into registers, compute tile k from SMEM, then `STS` the registers. Two `__syncthreads()` move to make that legal. On Ampere+ this becomes [[cp-async]], and on Hopper [[tma]].
- **Kernel 5:** with a swizzled layout, moving to the next mma tile isn't `ptr += const`. The fix is to XOR the per-thread address with a constant, which cuts ~13 integer ops per `ldmatrix` down to 1 XOR. Total instructions fell to ~90M, slightly below cuBLAS's 94,175,232. Kernel 4 had executed 216,227,840 instructions, 92% of them in the warp loop, mostly LOP/IADD3/SHF.
- **Kernel 6:** a second SMEM buffer removes one of the two `__syncthreads()` per iteration. CPI was 14, with ~2.6 cycles in sync stalls. The speedup is small.
- **Not done:** a coalesced epilogue (the current one writes uncoalesced and likely hurts small sizes), and SASS-level tuning.

## Numbers worth remembering (T4, as stated)
| Quantity | Value |
|---|---|
| Achievable GMEM bandwidth | 220 GB/s (68% of 320 theoretical) |
| L2 bandwidth used in balance calc | 1280 GB/s |
| Achievable SMEM bandwidth | ~3662 GB/s |
| HMMA.1688 measured | 49,439 GFLOP/s (76% of theoretical). 65,000 used as an overestimate |
| FFMA measured | 7,455 GFLOP/s (92%) |
| Balance: TC vs GMEM / L2 / SMEM | 224 / 38 / 13 FLOP/B |
| FFMA vs GMEM | ~33 FLOP/B (TCs are ~6.6× harder to feed) |
| Kernel 4 regs/thread | 166 (up from 104). ~49 KB SMEM → 1 block/SM |
| Final tiles (blog) | BM=BN=256, BK=32, "WM=WN=64". The code actually uses WM=128, WN=64 (see [[hidden-treasures-3-gemm-code]]) |

- The blog's own C/D-shape text for m16n8k8 says (8,8). It's 16×8, a typo in the post.
- The post says 64 KB SMEM per SM in one place and 62 KB in another.
- Per-kernel absolute TFLOP/s are only in images. The repo plot is transcribed in [[hidden-treasures-3-gemm-code]].

## Code patterns
```cuda
// A fragment for m16n8k8: 16x8 halves = two 8x8 matrices → ldmatrix .x2, 2 regs/thread
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
             : "=r"(a0), "=r"(a1) : "r"(smem_addr_of_row(lane % 16)));
// B fragment (8x8, stored row-major [K][N] in smem) → .trans gives the col-major fragment
asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
             : "=r"(b0) : "r"(smem_addr_of_row(lane % 8)));
// D = A*B + C, fp16 accumulate (Turing-era; for bf16 you MUST use .f32 accum)
asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};"
             : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
```
- Every `ldmatrix` lane supplies the address of **one 8-element (16 B) row**. Lanes 0–7 address matrix 0, lanes 8–15 matrix 1, and so on.
- The shared address must be a 32-bit shared-window address, converted with `cvta.to.shared` (or `__cvta_generic_to_shared`).

## Pitfalls
- FP16 accumulation (`.f16` C/D) is used throughout. It's fine for the benchmark, **wrong for the BF16 challenge**: bf16 `mma` only exists with `.f32` accumulators, per the [[mma-ptx]] PTX syntax.
- Swizzling moves the cost into index math. Precompute per-thread base addresses and step with XOR/add of constants.
- After ~70% of cuBLAS, profiling stopped pointing at a single bottleneck, and the rest was trial and error.
- Big tiles (256×256) hurt small problems. Per the repo plot, kernels 5/6 lose to kernel 4 at 1024³ (inference: too few CTAs to fill the GPU).

## Relevance to the serving track
- **Transfers (inference):**
  - The `ldmatrix` + `mma.sync` mechanics, fragment ownership, and the shared-address conversion. H100/H200 still execute `mma.sync` m16n8k16 for bf16. It doesn't reach wgmma peak, but at m ≤ 256 the kernel is HBM-bound, so `mma.sync` is a reasonable engine.
  - The SMEM swizzle/padding analysis, needed whenever W tiles are staged through SMEM.
  - The register-prefetch trick, a portable way to overlap the next W load with current math.
- **Does not transfer (inference):**
  - The whole intensity argument for BM=BN=256. With m ≤ 256 the **block tile in M is at most m**, and intensity is ≈ m FLOP per weight byte. We can't reach the compute roof, so tile size is chosen for **SM fill and bytes in flight**, not reuse.
  - The `.trans` B load. Our W is `[N,K]` with K contiguous, which is already the "col" layout the B operand wants, or the row-major A operand if you swap AB. No transpose is needed.
  - FP16 accumulate.
  - Turing-only workarounds: Hopper has [[cp-async]], [[tma]] and [[wgmma]].
- The uncoalesced-epilogue warning applies: at small m the C write is tiny, but split-K partial writes and reductions aren't → [[split-k-and-stream-k]].

## Related
[[hidden-treasures-3-gemm-code]] · [[mma-ptx]] · [[gemm-tiling-hierarchy]] · [[cp-async]] · [[swizzling]] · [[smem-bank-conflicts]] · [[siboehm-sgemm-worklog]] · [[bruce-lee-mma-ptx]] · [[gemv-and-skinny-gemm]] · [[roofline-and-arithmetic-intensity]] · [[wgmma]] · [[tma]]
