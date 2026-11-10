---
title: "Register-direct weight streaming into mma.sync fragments (k-permutation trick)"
type: concept
tags: [gpu, gemv, skinny-gemm, mma-ptx, swap-ab, small-m, memory-bound, cuda, negative-result, muse]
source_url:
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html   # m16n8k16 fragment layout (via [[mma-ptx]])
  - ../muse_bf16_proj/skinny_cuda.py                                    # the implementation measured here
---
# Register-direct weight streaming into mma.sync fragments

**TL;DR**
- **The idea** (option iii in [[gemv-and-skinny-gemm]]): skip shared memory and `ldmatrix`. Each thread loads W straight from global memory into registers with 16 B `ld.global.nc.L1::no_allocate` loads, already in `mma.sync m16n8k16` A-fragment order. This works because a dot product doesn't care about k order, as long as **W and x use the same k-permutation**.
- **Measured on H100 (2026-11-08): it lost to tuned Triton on every BF16 shape** in `muse_bf16_proj`. Best case was a tie (pooler m=16: 23.8 vs 24.3 µs). Worst was qkv_gate m=8: 17.3 vs 14.9 µs. It won only on the FP32 router at m=8 (5.44 vs 5.79 µs), where the hi/lo split happens in registers. Full numbers: [[findings-log]].
- **Profiled verdict** (Nsight Compute, below): it's latency-bound because in-flight bytes live in registers, which caps occupancy. Prefer a `cp.async`/TMA SMEM ring.

## Key ideas
### The permutation (thread (g, t), g = lane>>2, t = lane&3)
- A warp owns 16 W rows (fragment rows g and g+8) and processes 64-k blocks as 4 mma k-steps.
- The fragment only says *which logical k* a register holds. Map logical → actual k any way you like, if the map is a bijection per 64-k block and x's B fragment uses the same map.
- **v1/v2 map** (each thread owns 32 contiguous bytes of each row):

| logical k (step s) | actual k |
|---|---|
| 2t + j | 16t + 4s + j |
| 2t + 8 + j | 16t + 4s + 2 + j |

  That's 2 uint4 loads per row per thread. Registers for step s are uint4 `s/2`, components `2(s%2)` and `+1`.
- **v3 map** (for each load instruction, the 4 lanes of a row read 64 contiguous bytes):
  - BF16: uint4 v covers actual k `32v + 8t … +7`
  - FP32: uint4 v covers `16v + 4t … +3`
  - The register indexing is unchanged. x must follow the *element-type-specific* map, so the FP32 router needs a different x gather.
- B fragment: thread (g,t) holds token g. Tokens ≥ m are zero-filled, so m=1 uses one n8 tile and wastes 7/8 of a free MMA.
- FP32 weights: load f32, split in registers (`hi = rn(w)`, `lo = rn(w − hi)`), two MMAs into the same f32 accumulator. Same bytes as a stored hi+lo pair, and only one stream.

### Pipeline
- **v1:** issue all of one batch's loads, then compute. **Slow:** every batch costs a full memory round-trip per warp.
- **v2:** double-buffered registers (load batch i+1, compute batch i). This gave 17.2 → 15.9 µs on qkv_gate m=1. **Always double-buffer a register stream.**
- The epilogue is the same in all versions: intra-CTA reduction across warps through shared memory, then cross-CTA split-K with a self-resetting ticket ([[split-k-and-stream-k]]).

## Numbers worth remembering (H100 SXM, cold L2, graph replay, µs, best tuned config)
| shape | Triton tuned | CUDA v2 | CUDA v3 |
|---|---|---|---|
| qkv_gate m=1 | 12.9–13.2 | 15.9 | 17.7 |
| pooler m=1 | 22.7–22.9 | 23.2 | 25.9 |
| qkv_gate m=8 | 14.5–14.9 | 17.3 | 21.1 |
| pooler m=16 | 23.6–24.3 | 23.8 | 23.8 |
| router m=8 (FP32 W) | 5.8–5.9 | 6.0 | **5.44** |
| m=32 layers | — | 1.3–1.6× slower | 1.3–1.6× slower |

## Code patterns
```cuda
// v2/v3 main loop shape: two register batches, loads for one in flight while the other computes
B ba, bb;  ba.load(W, k0);
for (int kb = k0; kb < kend; kb += 2 * STEP) {
  if (kb + STEP < kend) bb.load(W, kb + STEP);
  ba.compute(acc, X, kb);
  if (kb + STEP >= kend) break;
  if (kb + 2 * STEP < kend) ba.load(W, kb + 2 * STEP);
  bb.compute(acc, X, kb + STEP);
}
```

## Nsight Compute verdict (2026-11-09, [[findings-log]])
- **The binding constraint is bytes in flight, not coalescing.**
  - Register-direct holds in-flight W in registers: 99 regs/thread → 2 CTAs/SM → ~2.1 warps per scheduler, 0.1 eligible, `long_scoreboard` 18–29 cycles per issued instruction.
  - Triton's m ≥ 8 kernel runs ~1 warp per scheduler and still wins, because its in-flight bytes sit in a 5-stage `cp.async` SMEM ring.
- v2 → v3 halved L1 load sectors (2.51 M → 1.25 M) and still got slower. The sector hypothesis below was wrong.
- **Follow-up v4 (`cp.async` ring + swizzle):** needs x in the ring too (see [[findings-log]]). It gets within 5–10% of Triton at m ≤ 16 but doesn't beat it. **Verdict: for BF16 weights, use Triton or an SMEM-ring kernel, not register-direct.**
- The register-direct path still pays off for FP32 weights at m=8 (router), where the in-register hi/lo split removes a second weight stream.

## Pitfalls
- **The m=32 collapse (inference):** x is re-gathered from L1 by every warp, NT=4 token tiles × 2 uint4 per 64-k block, so L1/LSU traffic for x ≈ m/16 × the W traffic. Stage x in shared memory once per CTA before going past m=16.
- **v3 got slower, not faster**, even though every load instruction now hits full sectors. Hypotheses (unverified):
  - The `L2::256B` prefetch hint interacts differently with the two layouts.
  - v2's per-thread contiguity let both loads of a thread hit the same sector, which the LSU merges.
  - More address math per load.
  Measure with `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld,dram__throughput` before guessing again.
- `asm volatile` on loads and MMAs pins the instruction order. That's good for keeping loads ahead of uses, but it stops the compiler interleaving.
- With these shapes, CTA count ≈ N/16 (e.g. 144 for qkv_gate). That's barely over one wave of 132 SMs, so add split-K or more rows per CTA to balance.

## Relevance to the serving track
- Tuned Triton (swap-AB `tl.dot` + split-K, CUDA-core GEMV at m=1) is the stronger baseline for dense BF16 skinny GEMMs at m ≤ 32 until this is profiled. Use [[triton]] first.
- The **in-register FP32 → hi/lo split** is worth keeping for the router and any FP32-weight GEMV. It reads the FP32 weight once and needs no repacked copy.

## Related
[[gemv-and-skinny-gemm]] · [[mma-ptx]] · [[memory-bandwidth-and-latency]] · [[split-k-and-stream-k]] · [[triton]] · [[findings-log]] · [[rorqual-h100-workflow]]
