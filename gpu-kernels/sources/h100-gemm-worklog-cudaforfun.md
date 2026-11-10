---
title: Outperforming cuBLAS on H100 – a Worklog (cudaforfun)
type: source
tags: [gpu, hopper, h100, gemm, wgmma, tma, warp-specialization, mbarrier, clusters, persistent, bf16]
source_url:
  - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
  - https://github.com/pranjalssh/fast.cu   # article's code (h100/matmul/matmul_{1..12}.cuh); snippets below are copied from it
---
# Outperforming cuBLAS on H100 – a Worklog (Pranjal Shankhdhar, Nov 2024)

**TL;DR**
- Step-by-step BF16 GEMM on H100 SXM from 32 TFLOPs (SIMT) to **764 TFLOPs vs cuBLAS 716** at M=N=K=4096 (~107%).
- The big jumps: tensor cores via [[wgmma]] + [[tma]] + mbarriers (32→317), bigger tiles (→423), [[warp-specialization]] producer/consumer (→498), 2 consumer warpgroups + `setmaxnreg` (→631), [[persistent-kernels-and-tile-scheduling]] with L2-grouped order (→660), raw PTX mbarriers (→704), [[thread-block-clusters]] + TMA multicast (→734), then micro-opts, TMA store, Hilbert order (→764).
- Register spills serialize wgmma: a 128×256 tile in one consumer warpgroup collapsed to 123 TFLOPs.
- The author states cuBLAS stays stronger on **small-K and matrix-vector** cases; gains at small N are limited because it's memory-bound. That's the regime our serving-track kernels live in.
- Most of the code is shown only as images in the article. The real source is the fast.cu repo, and the snippets here come from it.

## Key ideas (progression, M=N=K=4096 BF16, H100 SXM, CUDA 12.6)
Setup: A row-major, B and C column-major (so both operands are K-major). Inputs ~N(0,1). Average of 8 runs after 1 warmup. cuBLAS = **716 TFLOPs**.

| # | Change | TFLOPs | Why (per author) |
|---|---|---|---|
| 1 | Simon Boehm's SIMT kernel ported to bf16 | 32 | no tensor cores |
| 2 | `wgmma` + TMA + CUDA barriers; block tile = one instruction (64×64×16), 128 threads | 317 | tensor cores + async bulk loads |
| 3 | BM=128, BN=128, BK=64, `m64n128k16`, loop over sub-tiles | 423 | larger tiles amortize loads; larger wgmma N is faster |
| 4 | Producer/consumer warp specialization, circular SMEM queue QSIZE=5 | 498 | overlaps loads (~1415 cyc) with TC work (~703 cyc) |
| 5a | 128×256 tile, 1 consumer WG | **123** | register spill → wgmma serialized (ptxas warning C7511) |
| 5b | 2 consumer WGs + 1 producer (384 thr), QSIZE 5→3 for SMEM | 610 | halves per-thread accumulator regs (168 regs/thread) |
| 5c | `setmaxnreg`: consumers 240, producer 24 regs | 631 | 240·128·2 + 24·128 = 64512 ≤ 65536 |
| 6a | Persistent kernel (grid = #SMs), overlap tile i stores with tile i+1 loads | **400** | regression: naive tile order has poor L2 locality |
| 6b | Grouped tile order (16×8 tile groups) on 128 SMs | 660 | L2 hit rate 83% vs cuBLAS ~70% |
| 6c | Same on all 132 SMs | 655 | 132 doesn't divide the groups evenly |
| 7 | Hand PTX mbarrier (no `arrival_token`, manual parity) | 704 | token syncs cut ~257 → 3 |
| 8 | Clusters of 2 + TMA multicast + cluster-scope arrive | 734 | two SMs share one operand load; vertical (2×1) pairs best; larger clusters slower |
| 9 | Store reordering, `__stwt()` streaming stores, skip accumulator zeroing (wgmma scale-d=0 on first k) | 747 | diminishing returns |
| 10 | Stage C in SMEM, async TMA store (`cp.async.bulk.tensor ... global.shared::cta`) | 758 | offloads stores; costs queue SMEM |
| 11 | Hilbert-curve tile order | 764 | consecutive tiles stay spatially close across group boundaries |

Final kernel vs cuBLAS by square size: N=512 +2%, 1024 +17%, 2048 +7–8%, 4096 +7–8%, 8192 +1.5%. Small N is memory-bound and large N is power-bound (700 W cap).

Profiling cycles per block (kernel 4–6 era): load 1415, tensor core 703, **store 4572**. Stores dominated, which motivated persistent overlap and the TMA store.

## Numbers worth remembering (as stated)
- H100 SXM: 132 SMs, 4 tensor cores/SM, ≤1024 threads/SM, 65,536 regs/SM, ≤255 regs/thread, 256 KB combined SMEM+L1 per SM, 80 GB HBM3 @ **3.35 TB/s**, 50 MB L2 (the author notes it acts as ~2×25 MB partitions), 700 W.
- wgmma shapes: `m64nNk16`, N = 8…256. `m64n16k16` A+B operands = 2.5 KB SMEM. n=256 needs ~40 KB SMEM and 128 accumulator regs/thread.
- 64×N FP32 accumulator = 64·N/128 regs/thread (64×16 → 8 regs/thread).
- Final config (kernel 8/10): BM=128, BN=256, BK=64, 384 threads, QSIZE=3, cluster (2,1,1), grid = 128 CTAs.

## Code patterns (from fast.cu, `h100/matmul/matmul_8.cuh` / `_10.cuh`)
SMEM matrix descriptor for 128B-swizzled K-major bf16 tiles:
```cuda
__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }
__device__ uint64_t make_smem_desc(bf16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= matrix_descriptor_encode(addr);                 // start address >>4, bits 0-13
    desc |= matrix_descriptor_encode((uint64_t)16) << 16;   // leading byte offset (LBO)
    desc |= matrix_descriptor_encode((uint64_t)1024) << 32; // stride byte offset (SBO) = 8 rows * 128B
    desc |= 1llu << 62;                                     // swizzle mode = 128B
    return desc;
}
```
TMA tensor map as a 3-D view so each box row is exactly 64 bf16 = 128 B (matches SWIZZLE_128B):
```cuda
uint64_t gmem_prob_shape[5]  = {64, (uint64_t)global_height, (uint64_t)global_width/64, 1, 1};
uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width, 64*sizeof(bf16), 0, 0, 0};
uint32_t smem_box_shape[5]   = {64, uint32_t(BlockMajorSize), uint32_t(BlockMinorSize/64), 1, 1};
cuTensorMapEncodeTiled(&tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gmem_address, gmem_prob_shape,
    gmem_prob_stride, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
// kernel takes it as: const __grid_constant__ CUtensorMap tensorMapA
```
Producer loop (one thread of warpgroup 0) and consumer loop; full/empty mbarrier ring with parity `p`:
```cuda
// init (thread 0): full[i] count 1 (+tx bytes), empty[i] count num_consumers*CLUSTERS
if (wg_idx == 0) {                       // producer
  warpgroup_reg_dealloc<24>();            // setmaxnreg.dec.sync.aligned.u32 24
  if (tid == 0) for (...k...) {
    if (qidx == QSIZE) { qidx = 0; p ^= 1; }
    wait(&empty[qidx], p);                // mbarrier.try_wait.parity spin
    expect_bytes(&full[qidx], (BK*BN+BK*BM)*sizeof(bf16)); // mbarrier.arrive.expect_tx
    load_async(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], k*BK, m*BM);  // cp.async.bulk.tensor
    load_async(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], k*BK, n*BN);
  }
} else {                                  // consumers
  warpgroup_reg_alloc<240>();             // setmaxnreg.inc.sync.aligned.u32 240
  for (q...) arrive(&empty[q]);           // pre-release all slots
  for (...k...) {
    wait(&full[qidx], p);
    warpgroup_arrive();                   // wgmma.fence.sync.aligned
    for (k_it...) wgmma<BN,1,1,1,0,0>(d[m_it], &sA[...k_it*16], &sB[...k_it*16]);
    warpgroup_commit_batch();             // wgmma.commit_group.sync.aligned
    warpgroup_wait<0>();                  // wgmma.wait_group.sync.aligned 0
    arrive(&empty[qidx]);                 // give slot back
  }
}
```
TMA store epilogue (kernel 10): registers → SMEM `sC`, named barrier over the 256 consumer threads, one thread issues the store:
```cuda
asm volatile("cp.async.bulk.wait_group 0;");   // previous tile's store done reading sC
/* ... write d[][][] into sC ... */
asm volatile("bar.sync 10, 256;\n");           // consumers only (producer WG not included)
if (threadIdx.x == 128) {
  store_async(&tensorMapC, sC, m*BM, n*BN);    // cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group
  asm volatile("cp.async.bulk.commit_group;");
}
```
See [[wgmma]], [[warp-specialization]], [[thread-block-clusters]], and [[persistent-kernels-and-tile-scheduling]] for the cluster/multicast and scheduler code.

## Pitfalls
- Spilling serializes wgmma. Watch for ptxas warning C7511 ("Potential Performance Loss: wgmma.mma_async instructions are serialized").
- Going persistent without L2-aware ordering **lost** ~37% (640 → 400).
- Using all 132 SMs was slower than 128 because the tile groups don't divide 132 evenly. Wave/group quantization matters.
- Clusters larger than 2 were slower (inter-SM sync cost).
- The `cuda::barrier` arrival-token API adds syncs; raw PTX needs manual phase tracking (author: "abstraction leak").
- wgmma isn't in the CUDA C++ guide, so you write PTX. The author says the PTX docs' swizzle/core-matrix diagrams are wrong, and recommends letting TMA produce the swizzle.
- (inference) The kernel 10 snippet writes `sC` with generic stores and then issues a TMA store with no `fence.proxy.async.shared::cta`. The PTX memory model requires that fence between generic-proxy writes and async-proxy reads. Add it in your own code.
- (inference) Each consumer does `wgmma.wait_group 0` before releasing the slot, which leaves no wgmma in flight across k-blocks. `wait_group 1` and releasing the previous slot is the usual refinement.

## Relevance to the serving track
- The author says cuBLAS still wins on small-K and GEMV-like shapes, and that small N is memory-bound. For `muse_bf16_proj` (m ≤ 256) the compute-side tricks (Hilbert order, setmaxnreg rebalancing, 128×256 tiles) are mostly irrelevant. The **load pipeline** (TMA + mbarrier ring + a dedicated producer) is what carries over. (inference)
- (inference) Arithmetic intensity of x[m,k]·Wᵀ in BF16 is ≈ m FLOP/byte of weight. Even m=256 sits far below the H100 ridge point (~990 dense BF16 TFLOPs / 3.35 TB/s ≈ 300; the 990 figure is NVIDIA spec, not from this article). The goal is saturating HBM with deep TMA queues across all SMs, not TC utilization.
- (inference) Swap A/B: wgmma M is fixed at 64 per warpgroup, so put W rows on the M side and tokens on N (N = 8…256 in steps of 8). m=1/8/16 then wastes at most an n8 tile instead of a 64-row tile. See [[wgmma]].
- (inference) Cold L2 + each weight byte used by exactly one CTA means the L2-grouping/Hilbert wins (6b, 11) don't apply to weight streaming. Only x (tiny) is reused across CTAs.
- (inference) Store cost (4572 cycles here) is negligible when the output is m×n with small m. The persistent "overlap store with next load" trick matters less than overlapping the **prologue** (barrier init, descriptor fetch, first TMA latency) with the previous kernel. See [[pdl]] and [[cuda-graphs]].
- (inference) With 128 vs 132 SMs mattering at 4096³, tile counts vs 132 SMs matter even more for 2–17 µs kernels. Pick BN or split-K so that #work-units ≈ k·132.

## Related
[[wgmma]] · [[tma]] · [[warp-specialization]] · [[thread-block-clusters]] · [[persistent-kernels-and-tile-scheduling]] · [[tensor-core-generations]] · [[swizzling]] · [[smem-bank-conflicts]] · [[register-tiling]] · [[mma-ptx]] · [[pdl]] · [[cuda-graphs]] · [[colfax-blackwell-tmem-gemm]] · [[semianalysis-tensor-core-evolution]]
