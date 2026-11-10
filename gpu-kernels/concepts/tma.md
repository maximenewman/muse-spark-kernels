---
title: TMA (Tensor Memory Accelerator)
type: concept
tags: [gpu, hopper, sm90, tma, cp.async.bulk, tensormap, multicast, prefetch, memory-bound]
source_url:
  - https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html
  - https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html
  - https://yang-yifan.github.io/blogs/memory_model/memory_model.html
  - https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html   # cp.async.bulk.tensor, cp.async.bulk.prefetch.tensor, prefetch.tensormap, tensor modes
  - https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html   # cuTensorMapEncodeTiled constraints
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html   # alignment table, passing descriptors, swizzle
---
# TMA (Tensor Memory Accelerator)

**TL;DR**
- Hopper (sm_90+) async copy engine: one thread issues `cp.async.bulk.tensor` with a **tensor map (descriptor)** + integer **coordinates**; HW generates all addresses, handles OOB (zero/NaN fill), optional smem swizzle, and signals completion by **complete-tx bytes on an mbarrier** (loads) or **bulk async-group** (stores).
- Descriptor (`CUtensorMap`, 128 B, 64 B-aligned) is built on host with `cuTensorMapEncodeTiled` (or CuTe `make_tma_copy`) and passed as `const __grid_constant__` kernel param (recommended), `__constant__`, or global memory (needs `fence.proxy.tensormap`).
- Load path: `mbarrier.init` → `mbarrier.arrive.expect_tx(bytes)` → `cp.async.bulk.tensor...mbarrier::complete_tx::bytes` → `mbarrier.try_wait.parity`. Data is visible to generic proxy after the wait (TMA completion = release + implicit async-proxy fence).
- `cp.async.bulk.prefetch.tensor.L2` = same descriptor, to L2 only, no smem, no completion signal. `prefetch.tensormap` warms the descriptor itself.
- `.multicast::cluster` + 16-bit `ctaMask` writes the same box to the same smem offset in several CTAs of a cluster and signals each CTA's mbarrier.
- Hard constraints: gmem base 16 B-aligned, gmem strides multiple of 16 B, boxDim ≤ 256 per dim, box inner extent ×elemsize multiple of 16 B, smem dst 128 B-aligned (1024 B for 128B swizzle per programming-guide example), inner box bytes ≤ swizzle span.

## Key ideas
- **Why** (Yang): faster tensor cores + async copies left CUDA-core address generation/epilogue as the bottleneck; TMA offloads address generation, freeing CUDA cores for the epilogue. Only **1 thread** is needed to drive it.
- **Tensor modes** (PTX §5.5): *tiled* (layout preserved, box same rank as tensor) and *im2col* (box in DHW space, rows unrolled into columns for convolutions; rank ≥3). Blackwell adds `.tile::gather4/.scatter4` (4 arbitrary rows of a 2D tensor), `.im2col::w(::128)`. For GEMM/GEMV always use tiled.
- **Bounding box** (tiled): rank equals tensor rank; size multiple of 16 B; coordinates are `.s32` starting offsets; traversal stride (elementStrides ≤ 8) must be 1 in dim 0 for non-interleaved layouts. OOB: zero fill or OOB-NaN fill (`CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`). Load coords may be negative; store top-left may not.
- **Directions**: global→shared::cta (PTX 8.6+), global→shared::cluster (original Hopper form; required for remote mbarrier/multicast), shared::cta→global (`.bulk_group` completion: `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group.read 0` before reusing smem). Store-side reductions exist (not covered here).
- **Completion semantics**: complete-tx on the mbarrier has **.release at .cluster scope**; the copy itself is a weak memory op. The instruction accesses its mbarrier in the generic proxy; the tensormap is accessed in the **tensormap proxy**.
- **L2 cache hint**: `.L2::cache_hint` + 64-bit `cache_policy` (hint only). CuTe exposes it via `.with(mbar, mask, cache_hint)`. Tensor map also has `l2Promotion` (NONE/64B/128B/256B: L2 fill granularity from DRAM).
- **Multicast** (Yang Sec. 5): CuTe splits the cluster's tile along the **slowest-moving** dim so each CTA issues one contiguous slice (avoids being TMA-issue-bound in one CTA); each destination CTA's barrier is credited with the full tile's bytes, so no extra cluster sync after the wait. Requires `cluster_sync()` + `fence.mbarrier_init.release.cluster` after barrier init. PTX: multicast "advised" for sm_90a/sm_100a-family targets; programming guide: optimized only for local-GPU gmem.
- **Swizzle compatibility** (Yang mma_swizzle §9): TMA writes each box row contiguously into smem, then applies the 16 B-chunk swizzle. For an MMA-compatible layout the **box inner width must equal exactly one swizzle atom width** (16B/32B/64B/128B); height can be any multiple of 8. A 64×256 B tile with 128B swizzle ⇒ CuTe emits 2 TMA instructions (max box 64×128 B). Stacking atoms column-major (along M) allows taller boxes ⇒ fewer TMA instrs. Details → [[swizzling]].
- **Coordinates not pointers** (CuTe): `tma.get_tma_tensor(shape)` gives an ArithTuple coordinate tensor; tile/partition it the same way as the smem tensor so coordinates stay consistent.
- **Descriptor on device**: tiled maps can be modified in-kernel via `tensormap.replace` (sm_90a-specific) in smem, then `tensormap_copy_fenceproxy` to global; useful for grouped/batched problems.

## Numbers worth remembering
| Constraint (cuTensorMapEncodeTiled / PG Table 26) | Value |
|---|---|
| tensorMap object alignment | 64 B (object is 128 B; PG uses `ptx::n32_t<128>` for its size) |
| tensorRank | 1..5 (≥3 if interleaved) |
| globalAddress | 16 B aligned (32 B if INTERLEAVE_32B) |
| globalDim | 1..2^31 |
| globalStrides (bytes, dims 1..rank-1) | multiple of 16, < 2^40 |
| boxDim[i] | 1..256 elements; boxDim[0]·elemsize multiple of 16 B |
| elementStrides | 1..8 |
| inner box bytes with swizzle | ≤32 / ≤64 / ≤128 for SWIZZLE_32B/64B/128B |
| smem destination | 128 B aligned; PG swizzle example uses `alignas(1024)` for 128B swizzle ("aligned to the swizzle repeat") |
| mbarrier | 8 B aligned `.b64` in shared |
| transfer size | multiple of 16 B |
| im2col channelsPerPixel ≤256, pixelsPerColumn ≤1024 | |
- Swizzle granularity fixed at 16 B chunks; offset formula when smem not aligned to repeat: `(smem_ptr/128) % {8,4,2}` for 128B/64B/32B.

## Code patterns
```cpp
// Host: encode a 2D tiled map for W[n,k] bf16 row-major (dim0 = k, contiguous)
CUtensorMap tm;
cuuint64_t gdim[2]    = {K, N};
cuuint64_t gstride[1] = {K * sizeof(bf16)};          // bytes, must be %16==0
cuuint32_t box[2]     = {64, BN};                    // 64 bf16 = 128 B inner = one 128B swizzle atom
cuuint32_t estr[2]    = {1, 1};
cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, W, gdim, gstride, box, estr,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
// get the driver fn via cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", ..., 12000, ...)
```
```ptx
// Device (one elected thread). smem dst 1024B-aligned for 128B swizzle.
mbarrier.init.shared::cta.b64              [bar], 1;                 // once, then fence/sync
mbarrier.arrive.expect_tx.shared::cta.b64  _, [bar], 16384;         // bytes this phase will receive
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
                                           [sdst], [tmap, {k0, n0}], [bar];   // coords = {dim0, dim1}
// all consumers:
// loop: mbarrier.try_wait.parity.shared::cta.b64 p, [bar], phase;  @!p bra loop;
cp.async.bulk.prefetch.tensor.2d.L2.global.tile [tmap, {k1, n0}];   // L2 warmup, no barrier
prefetch.param.tensormap [tmap];  // or generic: prefetch.tensormap [addr]; warms the descriptor
```
```cpp
// CUDA C++ (libcu++) equivalent from the Programming Guide
ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, &smem, &tmap, coords,
                          cuda::device::barrier_native_handle(bar));
token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem));   // arrive + expect_tx
bar.wait(std::move(token));
// store back: write smem -> ptx::fence_proxy_async(ptx::space_shared); __syncthreads();
//   elected: cp_async_bulk_tensor(global<-shared); cp_async_bulk_commit_group(); cp_async_bulk_wait_group_read(0)
```
CuTe version: see [[yangyifan-cute-tma]].

## Pitfalls
- expect_tx bytes must equal the **full box bytes** including OOB-filled elements (the box is always fully written) (inference from "size of transfer"; verify on edge tiles).
- Swizzle mode vs box inner width mismatch → layout wrong for wgmma, or encode error (inner bytes > span).
- Smem written by threads and then TMA-stored requires `fence.proxy.async.shared::cta` before the barrier/sync (generic→async proxy). TMA-loaded data read by threads needs only the mbarrier wait.
- Descriptor in global memory modified at runtime needs `fence.proxy.tensormap::generic.acquire` before use; slower than grid-constant.
- Remote mbarrier (another CTA) requires `.shared::cluster` destination form; `.shared::cta` with a remote barrier is UB.
- Hopper `.multicast::cluster` should be compiled for `sm_90a`; generic `sm_90` "may have substantially reduced performance".
- gmem row pitch not a multiple of 16 B (e.g. k·2 B with k odd·…) cannot be described → pad or fall back to `cp.async`/`ld.global`.

## Relevance to the serving track
- W[n,k] bf16 row-major: dim0 = k. All Muse layer k's are presumably multiples of 8 elements → 16 B stride OK; check each layer's k and n. (inference)
- Small m: the x tile is m×BK (m∈{1,8,16,...}); boxDim[1]=m works down to 1, but a 1-row box of 128 B is a tiny transaction — for m=1 a plain vectorized `ld.global` of x (or one box covering all of k, ≤256 elements per dim) may be cheaper. (inference)
- Weight streaming wants many bytes in flight: multi-stage smem ring of W boxes (e.g. BN×64 bf16 = BN·128 B per stage) with one mbarrier per stage; issue all initial stages in the prologue. (inference)
- Cold L2: issue `cp.async.bulk.prefetch.tensor.L2` for the CTA's later k-tiles right at kernel start — and with [[pdl]], before `griddepcontrol.wait`, because W never depends on the previous kernel. (inference)
- Multicast is for operands shared across CTAs (x), not W. At m≤32 the x tile is small and L2 LRC already de-dups (see [[yangyifan-register-tiling]]); multicast + clusters add launch constraints, so measure before using. (inference)
- Router layer FP32 as BF16 hi+lo: two W tensors (hi, lo) → two descriptors or one 3D map with a size-2 outer dim; both stream through the same pipeline. (inference)
- Descriptors are kernel params ⇒ baked into the CUDA graph at capture; changing the W pointer requires re-capture or global-memory descriptors. (inference; see [[cuda-graphs]])

## Related
[[mbarrier]] [[pdl]] [[swizzling]] [[smem-bank-conflicts]] [[wgmma]] [[warp-specialization]] [[cute-layouts]] [[register-tiling]] [[cuda-graphs]] [[tensor-core-generations]] [[yangyifan-cute-tma]]
