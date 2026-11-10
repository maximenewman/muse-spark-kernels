---
title: tcgen05 and Tensor Memory (TMEM) – Blackwell sm_100
type: concept
tags: [gpu, blackwell, sm100a, tcgen05, tmem, umma, tensor-cores, nvfp4]
source_url:
  - https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
  - https://github.com/pranjalssh/fast.cu  (gb300/nvfp4/gemm9.cuh, real tcgen05 PTX)
---
# tcgen05 + TMEM (Blackwell datacenter, CC 10.x)

**TL;DR**
- `tcgen05.mma` (CUTLASS "UMMA") is issued by **one thread**, runs async, takes A from SMEM or TMEM and B from SMEM, and accumulates into **TMEM**.
- TMEM = 128 lanes × 512 columns × 32-bit = **256 KB/SM**. It's explicitly managed: `tcgen05.alloc` / `dealloc` / `relinquish_alloc_permit`, moved with `tcgen05.ld` (→RF), `tcgen05.st` (RF→), `tcgen05.cp` (SMEM→).
- Completion: `tcgen05.commit` makes an mbarrier arrive once all prior MMAs by this thread finish.
- `cta_group::2` = 2-SM MMA across a CTA pair (one TPC). M doubles and each SM loads half of B.
- **Not available on Hopper (H200) or consumer Blackwell (sm_120).**

## Key ideas
- **Shapes (1-CTA dense):** M=64 (N%8==0) or M=128 (N%16==0), N ≤ 256, K = 32 bytes. [colfax] 2-SM doubles M. [semianalysis]
- **Instruction:** `tcgen05.mma.cta_group::{1,2}.kind::{f16,tf32,f8f6f4,...} [d_tmem], a_desc|[a_tmem], b_desc, idesc, enable_input_d;`. `idesc` (32-bit) encodes dtypes, transpose/negate, sparsity, M, N. [colfax] Block-scaled kinds (e.g. `kind::mxf4nvf4.block_scale.scale_vec::4X`) also take TMEM addresses of the scale factors (fast.cu gb300).
- **TMEM addressing:** lane in bits 31:16, column in bits 15:0. Allocation in power-of-2 columns ≥ 32, all 128 lanes. [colfax]
- **Access restriction:** warp (w % 4) of a warpgroup may only access lanes 32·(w%4)…+31. Reading a full 128-row accumulator needs 4 warps. [colfax]
- **Exclusivity:** only tcgen05 ops touch TMEM. Pre/post-processing happens in RF/SMEM. [colfax]
- **Why TMEM:** D is accessed ~2·K_t times per tile. Keeping it out of the RF frees registers and lets the epilogue overlap the next tile's MMAs (double-buffered accumulators; 128×256 FP32 = 256 cols = half of TMEM). [semianalysis, colfax]
- **CTA pairs:** CTAs whose cluster rank differs in the last bit pair onto a TPC. The leader issues; B halves exchange over DSMEM. Effective SMEM per MMA ~doubles. [semianalysis]
- **Formats:** MXFP8/6/4, NVFP4 (block-16, E4M3 scale, plus a second-level FP32 scale). FP8 = FP6 throughput. 4:8 pair sparsity for NVFP4. [semianalysis]

## Numbers worth remembering
- TMEM 256 KB/SM (= RF size), 128 × 512 × 4 B. [colfax, semianalysis]
- Alloc: 32…512 columns, power of 2. [colfax]
- `tcgen05.ld` shapes: `.16x64b .16x128b .16x256b .32x32b`, `.num` = x1…x128. Max ≈ 128 regs/thread per instruction. [colfax]
- fast.cu gb300 notes K=64 (and sm_103a-only K=96) NVFP4 MMA selected by idesc bit 31 (repo comments, not Colfax).

## Code patterns
Lifecycle (real PTX from fast.cu gb300/nvfp4, 2-CTA variants; drop `cta_group::2` → `::1` for 1-SM):
```cuda
// 1. one warp allocates; base TMEM address is written to smem
asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_slot), "r"(n_cols));
asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");   // let other CTAs alloc
// 2. one elected thread issues MMAs (acc=0 on first k-step overwrites D)
"@p tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], acc;"
//      d_tmem, a_desc, b_desc, idesc, sfa_tmem, sfb_tmem, accumulate-predicate
// 3. signal completion to an mbarrier (multicast to both CTAs of the pair)
"@p tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
// 4. epilogue warps wait on that mbarrier, then read TMEM -> registers
asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,...,%31}, [%32];" : ... : "r"(taddr));
asm volatile("tcgen05.wait::ld.sync.aligned;");    // registers valid only after this
// 5. same warp that allocated frees
asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(n_cols));
```
CuTe equivalents [colfax]: `SM100_MMA_F16BF16_SS` / `_TS` atoms, `TMEM::Allocator1Sm{}.allocate(512, &tmem_base)`, `umma_arrive(&bar)` (= tcgen05.commit), `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc)` (4 warps), `ScaleOut::Zero` then `ScaleOut::One`.

## Pitfalls
- Reading TMEM or overwriting SMEM operands before the commit→mbarrier wait is a race. [colfax]
- `cute::gemm` called from one thread deadlocks (CUTLASS elects inside a warp). [colfax]
- Alloc and dealloc in the same warp. A CTA that grabs 512 columns and doesn't relinquish blocks co-resident CTAs. [colfax]
- Use `tcgen05.wait::ld` before consuming `tcgen05.ld` results, and don't release the accumulator buffer before it (fast.cu comments fuse wait+arrive for this reason).
- Debug: `nvcc --g-tensor-memory-access-check` (as named by Colfax).

## Relevance to the serving track
- (inference) **Doesn't apply to the H200 target.** Keep it for reading SGLang/FlashInfer/CUTLASS sm100 kernels (NVFP4 MoE, attention) and for any future B200/GB300 problem. If a reference kernel is tcgen05-based, the Hopper port must be rewritten around [[wgmma]] (accumulator in RF, warpgroup issue, no FP4 MMA → dequant NVFP4 to BF16/FP8 first).
- (inference) On Blackwell, small-m decode maps tokens to N (N%8 at M=64, N%16 at M=128) and weights to M. 2-SM mode helps only if B (tokens) traffic matters, which it doesn't at small m. Weights are on the A side, unique per CTA.

## Related
[[tensor-core-generations]] · [[wgmma]] · [[warp-specialization]] · [[thread-block-clusters]] · [[tma]] · [[cute-layouts]] · [[colfax-blackwell-tmem-gemm]] · [[semianalysis-tensor-core-evolution]]
