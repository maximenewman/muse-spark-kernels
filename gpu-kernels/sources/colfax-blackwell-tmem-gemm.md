---
title: Colfax – CUTLASS Tutorial, GEMM with Tensor Memory on Blackwell
type: source
tags: [gpu, blackwell, sm100, tcgen05, tmem, umma, cutlass, cute, gemm]
source_url:
  - https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
---
# Colfax – Writing GEMM Kernels Using Tensor Memory for NVIDIA Blackwell GPUs (Part 1)

**TL;DR**
- Datacenter Blackwell (CC 10.0, sm_100a) replaces Hopper `wgmma` with **`tcgen05.mma`** (CUTLASS calls it "UMMA"). It's issued by **one thread**, A comes from SMEM or TMEM, B from SMEM, and the accumulator D always lives in **TMEM**.
- TMEM: 256 KB/SM = **128 lanes × 512 columns × 32-bit**. It's explicitly allocated (`tcgen05.alloc`, power-of-2 columns ≥ 32), freed (`tcgen05.dealloc`, same warp), and read out with `tcgen05.ld` into registers.
- Warp *w%4* of a warpgroup can only touch TMEM lanes 32·(w%4)…+31, so the epilogue needs a full warpgroup.
- In CuTe the MMA atom becomes CTA-level (`ThrID = Layout<_1>`). Thread-mode layouts are repurposed for CTA peers (2-SM mode, covered in Part 2).
- **No performance numbers** in this article. Part 1 is purely about the programming model.
- Consumer Blackwell (CC 12.0) has **no TMEM**.

## Key ideas (article progression)
1. **UMMA vs WGMMA.** WGMMA is not supported on Blackwell. New in tcgen05: FP4/FP6, higher throughput, built-in block scaling, TMEM accumulators, CTA-pair operation, single-thread launch. It's async; D = A·B or D = A·B + D.
2. **PTX form:** `tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc, enable-input-d;`. Kinds: `.kind::f16`, `.kind::tf32`, `.kind::f8f6f4` (also block-scaled kinds). `.cta_group::1|2`.
   - a-desc/b-desc: 64-bit SMEM matrix descriptors (address, LBO/SBO, swizzle), like WGMMA. Tiles are expected K-major (transposes are allowed via idesc). If A is in TMEM, a-desc is replaced by a TMEM address.
   - `idesc`: 32-bit instruction descriptor holding dtypes, sparsity, transpose/negate for A/B, and M/N.
   - `enable-input-d`: predicate. False → D = A·B (zero-init), true → accumulate.
3. **Shapes (dense, 1-CTA):** M=64 with N%8==0, or M=128 with N%16==0. N ≤ 256. K = 32 bytes (16 elements for 16-bit types). The largest atom, 128×256×16 FP32 accumulate, uses exactly **half** of TMEM (256 of 512 columns), so you can double-buffer accumulators.
4. **TMEM.** Address = lane in bits 31–16, column in bits 15–0 (lane 1, col 1 → `0x00010001`). Allocation is in whole columns spanning all 128 lanes. `alloc` writes the base address into SMEM. `tcgen05.relinquish_alloc_permit` promises no further allocation so other CTAs can be scheduled on the SM. Data in: UMMA result, `tcgen05.cp` (SMEM→TMEM), `tcgen05.st` (RF→TMEM). Out: `tcgen05.ld` (TMEM→RF). Nothing else can read TMEM, so any pre/post-processing happens outside.
5. **`tcgen05.ld`** shapes `.16x64b`, `.16x128b`, `.16x256b`, `.32x32b`, repeat `.x1`…`.x128`. Warp-wide `.sync.aligned`. Larger `.num` means fewer instructions but more registers (cap ≈ 128 regs/thread per instruction). The accumulator is row-major in TMEM (row = lane), so no fragment thread-value layout is needed.
6. **CuTe interface.** Atom `SM100_MMA_F16BF16_SS<TA,TB,TC,M,N,majorA,majorB,...>` (`SS` = A,B in SMEM; `TS` = A in TMEM). Size and dtypes move from the name into template params. `ThrMMA` becomes `cta_mma` via `tiled_mma.get_slice(mma_v)` (mma_v = peer-CTA index, always 0 in 1-SM mode).
   - SMEM layout: `UMMA::Layout_K_SW128_Atom<TypeA>` tiled by `UMMA::tile_to_mma_shape`.
   - `make_fragment_A/B` return **descriptor iterators**, not registers.
   - Accumulator tensor `tCtAcc` has lane stride 65536 (= `1<<16`).
   - Example tiler: 128×256×16 atom, bK = 4 MMAs, so 4 UMMAs per GMEM→SMEM copy.
   - `cute::TMEM::Allocator1Sm` with `allocate`/`free`. The example allocates all 512 columns.
7. **Sync.** One thread inits mbarrier (count 1), then `__syncthreads()`. An elected thread in one warp issues all UMMAs for the k_block (first with `ScaleOut::Zero`, then `ScaleOut::One`), then `umma_arrive(&mma_barrier)` (= `tcgen05.commit` → mbarrier arrive). All warps `wait_barrier(mma_barrier, phase)` and toggle phase each iteration.
8. **Epilogue.** `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc)` (wraps `tcgen05.ld.sync.aligned.32x32b.x1.b32`; "hardcoded to use 4 warps, or 1 warpgroup"). Then `partition_S` on TMEM, `partition_D` on GMEM, `make_tensor<AccType>` registers, copy, `axpby`, store. Thread 0 shapes: `tDtAcc ((_32,_1),_256,_1,_1)`, `tDrAcc ((_1,_1),_256,_1,_1)`.

## Numbers worth remembering
- TMEM 256 KB/SM = 128 lanes × 512 cols × 4 B (same as the register file size, per [[semianalysis-tensor-core-evolution]]).
- Alloc granularity: power-of-2 columns, min 32, max 512.
- UMMA 1-CTA: M ∈ {64, 128}, N ≤ 256 (multiples of 8 / 16), K = 32 B per instruction.
- 128×256 FP32 accumulator = 256 TMEM columns = half of TMEM.
- 128B swizzle atom = 16 × 4 × 2 B (the article's arithmetic for 16-bit K-major).

## Code patterns
Real PTX of the same family (from fast.cu `gb300/nvfp4/gemm9.cuh`, a 2-SM NVFP4 kernel, not from the Colfax article):
```cuda
// allocate n_cols TMEM columns (2-CTA pair variant); base address written to smem
asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_addr), "r"(n_cols));
asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
// widest TMEM read: each lane-thread gets 32 consecutive 32-bit columns
asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,...,%31}, [%32];" : ... : "r"(taddr));
// MMA completion -> mbarrier arrive (multicast to both CTAs of the pair)
"@p tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(n_cols));
```
CuTe skeleton (as described in the article, paraphrased):
```cpp
TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TA,TB,TAcc,128,256,UMMA::Major::K,UMMA::Major::K>{});
auto cta_mma = tiled_mma.get_slice(0);                 // peer-CTA index (0 in 1-SM mode)
Tensor tCrA = cta_mma.make_fragment_A(tCsA);           // smem descriptors, not registers
Tensor tCtAcc = cta_mma.make_fragment_C(...);          // TMEM tensor, lane stride 1<<16
// alloc: TMEM::Allocator1Sm().allocate(512, &shared.tmem_base_ptr)
if (elect_one_warp) { for (k_block...) gemm(tiled_mma, tCrA(_,_,k), tCrB(_,_,k), tCtAcc); umma_arrive(&shared.mma_barrier); }
wait_barrier(shared.mma_barrier, phase); phase ^= 1;
auto t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);   // 4 warps
```

## Pitfalls
- `gemm(...)` takes no barrier. Overwriting A/B SMEM or reading TMEM before the `umma_arrive` → mbarrier wait completes is a race. Editing the descriptor tensor (`tCrA`) is safe; editing the SMEM tensor (`tCsA`) is not (author reply to a comment).
- Calling `cute::gemm` from a single thread deadlocks. CUTLASS elects internally, so call it from a whole warp.
- Alloc/dealloc must be the same warp, with power-of-2 columns ≥ 32. A CTA holding 512 columns blocks any co-resident CTA from allocating. Use `relinquish_alloc_permit`.
- Runtime TMEM checker: the article names `nvcc --g-tensor-memory-access-check`.
- sm_120 (RTX/consumer Blackwell) has no TMEM or tcgen05, so this design doesn't port.

## Relevance to the serving track
- (inference) **Target is H200 (Hopper sm_90a): none of tcgen05/TMEM exists there.** This note matters only for (a) understanding Blackwell-oriented reference kernels (FlashInfer/CUTLASS sm100 NVFP4 MoE) you might be tempted to port, and (b) future Blackwell problems. On H200, NVFP4 weights must be dequantized in software to BF16/FP8 before `wgmma`. Hopper has no FP4 tensor-core path.
- (inference) For small-m on Blackwell the shape rules matter: UMMA M ∈ {64,128} on the "A" side. As with Hopper, put weights on M and tokens on N (N%8 at M=64, N%16 at M=128).
- (inference) Single-thread issue + TMEM accumulators free the RF. A tiny epilogue (m×n output) is a few `tcgen05.ld`s. But `alloc` + `relinquish` + `dealloc` + mbarrier init add fixed prologue cost that's visible in 2–17 µs kernels.

## Related
[[tcgen05-tmem]] · [[tensor-core-generations]] · [[wgmma]] · [[warp-specialization]] · [[thread-block-clusters]] · [[cute-layouts]] · [[swizzling]] · [[tma]] · [[semianalysis-tensor-core-evolution]] · [[h100-gemm-worklog-cudaforfun]]
