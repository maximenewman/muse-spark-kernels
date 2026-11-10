---
title: "Yifan Yang – Using TMA Load, Prefetch, and Multicast Load in CuTe"
type: source
tags: [gpu, hopper, sm90, tma, cute, cutlass, multicast, prefetch, mbarrier]
source_url: https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html
code_url: https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/cute_tma/code
---
# Yifan Yang – Using TMA Load, Prefetch, and Multicast Load in CuTe

Author: Yifan Yang (ex-NVIDIA DL architect, now Anthropic; author of FlashInfer's `tgv_gemm` low-latency GEMM and TRT-LLM's PDL work). Fetched in full 2026-11-08.

**TL;DR**
- TMA (Hopper) exists to offload **address generation** from CUDA cores so the address-gen/epilogue stages keep up with faster tensor cores and memory (Little's law argument).
- CuTe path: host `make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout)` builds the descriptor; device: 1 thread inits an mbarrier, sets expected tx bytes, `copy(tma.with(mbar), partition_S(coord_tile), partition_D(smem))`, everyone `wait_barrier(mbar, phase)`.
- **Prefetch** = same host code, device calls `prefetch(tma, partition_S(...))`; goes gmem→L2 only, no smem, no mbarrier, no completion signal.
- **Multicast** = `SM90_TMA_LOAD_MULTICAST` + cluster size; CuTe splits the tile along the **slowest-moving dim** across CTAs of the cluster, each CTA issues its slice, data + tx-bytes land in **every** CTA's smem/mbarrier selected by a 16-bit `ctaMask`.
- The TMA descriptor must be a `__grid_constant__` kernel param (built on host, immutable on device).
- Coordinates, not pointers: `get_tma_tensor(shape)` returns an ArithTuple "coordinate tensor" you tile/partition exactly like the smem tensor so coords stay in sync.

## Key ideas
- **Why TMA (Sec. 1)**: Volta path DRAM→L2→L1→RF→smem→RF→TC used RF/L1 as staging. Ampere `cp.async` bypasses RF/L1 (DRAM→L2→smem). With faster TCs, the remaining bottleneck is CUDA-core throughput for address generation + epilogue. TMA generates addresses itself at high throughput, leaving CUDA cores for the epilogue.
- **Ways to use TMA**: raw CUDA driver API (low level, error-prone), Triton (experimental at time of writing, less control), CuTe (high-level with low-level control). Update: CuTe DSL example uses the newer `tma_partition` API (functionally equivalent).
- **Host side** (Sec. 3.2): gmem layout `make_layout(make_shape(N,K), make_stride(K,1))` (row-major [N,K] — i.e. a weight matrix stored the way `x @ W.T` wants it); smem layout must use **static** ints (`Int<CTA_N>{}`), dynamic ints → compile error. Launch with blockDim 32 — "TMA only needs 1 thread to drive".
- **Device side** (Sec. 3.3): `__shared__ uint64_t mbar`; thread 0: `initialize_barrier(mbar, 1)` + `set_barrier_transaction_bytes(mbar, CTA_N*CTA_K*sizeof(T))`; build coord tile with `local_tile(gmem_coord, Tile<...>{}, make_coord(blockIdx.x, blockIdx.y))`; `get_slice(0)` (single CTA, no cluster); `copy(...)`. Then `__syncthreads(); wait_barrier(mbar, 0);` — wait completes when phase 0 flips, i.e. TMA decremented tx-count to 0.
- **Copy_Atom hierarchy** (Sec. 3.4.1): CopyOp → Copy_Traits → Copy_Atom → TiledCopy. For TMA a Copy_Atom = one CTA's TMA op; TiledCopy = all TMA ops of a cluster. Without cluster, atom size == TiledCopy size == whole CTA tile.
- **Non-executable vs executable atom**: the TMA atom has no mbarrier until `.with(mbar[, mcast_mask])`; `.with` can also embed an L2 cache hint.
- `partition_S/D` reshape [CTA_N,CTA_K] → [[ATOM_N,ATOM_K], CTA_N/ATOM_N, CTA_K/ATOM_K]; in practice atom == whole tile, so shape [[ATOM_N,ATOM_K],1,1] → one TMA instruction.
- **Dispatch chain** (Sec. 3.4.3): `copy()` → `copy_if()` (loops over atoms) → `Copy_Atom::call()` → `copy_unpack()` → `CallCOPY()`/`explode()` → `SM90_TMA_LOAD::copy()` → `SM90_TMA_LOAD_2D::copy()` → `cp.async.bulk.tensor.2d` PTX. Coordinates come from `src.data().coord_` (ArithTuple).
- **Prefetch** (Sec. 4): CuTe substitutes CopyOp `SM90_TMA_LOAD_2D` → `SM90_TMA_LOAD_2D::PREFETCH`; `copy_unpack` drops the dst pointer; lowers to `cp.async.bulk.prefetch.tensor`. "Useful in cases where the data loading latency is exposed." PTX has no arrival signaling for prefetch.
- **Multicast** (Sec. 5): launched with `cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension`. Needs `__syncthreads(); cluster_sync(); fence_barrier_init();` after barrier init so remote CTAs see initialized barriers before anyone multicasts into them. Tile indexed by `cluster_id_in_grid()`, slice by `block_rank_in_cluster()`. Mask `(1<<cluster_size)-1`; mask `0b01` writes only CTA 0.
- **Work split for multicast**: each CTA issues 1/cluster_size of the tile (so no single TMA unit is issue-bound), sliced along slowest-moving dim (keeps each chunk contiguous); if that dim is too small, continue onto next-slowest dim ("slicing the inverse of the smem layout", logic in `tma_partition()`).
- **No cluster_sync needed after load**: TMA updates tx-bytes on **all** destination CTAs' barriers, so each CTA's local barrier tracks completion of the whole tile, not just its own slice.

## Numbers worth remembering
- Harness example: FP8 e4m3 [256,256] matrix, TMA tile 64×128 (CTA_N×CTA_K). blockDim = 32.
- Multicast mask is 16-bit (`uint16_t`) in the CuTe example.

## Code patterns
```cpp
// host
auto gmem = make_tensor(make_gmem_ptr(W), make_layout(make_shape(N,K), make_stride(K,1)));   // [N,K] row-major
auto sL   = make_layout(make_shape(Int<BN>{},Int<BK>{}), make_stride(Int<BK>{},_1{}));      // static smem layout
auto tma  = make_tma_copy(SM90_TMA_LOAD{}, gmem, sL);                                        // builds CUtensorMap
kernel<<<grid, 32>>>(tma, gmem, sL);

// device
__global__ void k(__grid_constant__ const TmaLoad tma, G g, S sL) {
  __shared__ T smem[BN*BK]; __shared__ uint64_t mbar;
  auto sT = make_tensor(make_smem_ptr(smem), sL);
  if (threadIdx.x == 0) {
    initialize_barrier(mbar, 1);                          // mbarrier.init, arrive count 1
    set_barrier_transaction_bytes(mbar, BN*BK*sizeof(T)); // arrive + expect_tx(bytes)
    auto gC = local_tile(tma.get_tma_tensor(shape(g)), Tile<Int<BN>,Int<BK>>{}, make_coord(blockIdx.x, blockIdx.y));
    auto t  = tma.get_slice(0);
    copy(tma.with(mbar), t.partition_S(gC), t.partition_D(sT));   // cp.async.bulk.tensor.2d
  }
  __syncthreads();
  wait_barrier(mbar, 0);   // try_wait.parity until phase 0 completes
}
// prefetch variant: prefetch(tma, t.partition_S(gC));  // -> cp.async.bulk.prefetch.tensor, no mbar
// multicast variant: make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, gmem, sL, Int<CLUSTER>{});
//   copy(tma.with(mbar, mask), tma.get_slice(block_rank_in_cluster()).partition_S(gC_cluster), ...partition_D(sT));
```

## Pitfalls
- smem layout with dynamic ints → compile error.
- TMA object must be `__grid_constant__ const` (descriptor can't be modified on device in this path).
- Multicast: forgetting `cluster_sync()` + `fence_barrier_init()` after mbarrier init → remote CTAs may signal an uninitialized barrier.
- Prefetch gives no completion signal; don't rely on it for correctness.
- Multicast with a single "leader" CTA issuing the whole tile would make you TMA-issue-bound — CuTe deliberately splits it.

## Relevance to the serving track
- `muse_bf16_proj` W is [n,k] row-major (k contiguous) — exactly this blog's `[N,K]` row-major gmem example; tiles of W along k are the TMA box. (inference)
- Only one thread issues TMA → a 1-warp producer can stream W while other warps compute; for small m the whole CTA can be tiny, reducing prologue cost. (inference)
- **Prefetch to L2** is the cheapest way to raise memory-level parallelism under a cold L2: issue `cp.async.bulk.prefetch.tensor` for later k-tiles of W at kernel start (or before `griddepcontrol.wait`, since W does not depend on the previous kernel – see [[pdl]]). (inference)
- **Multicast is for shared operands**: x (m≤256 rows) is reused by every CTA — multicast x within a cluster rather than W (W tiles are unique per CTA in a skinny GEMM). But x is tiny; L2 already coalesces duplicate requests (see [[yangyifan-register-tiling]] GEMV/LRC section), so multicast of x likely only matters at m=128/256. (inference)
- Descriptor creation (`make_tma_copy` → `cuTensorMapEncodeTiled`) happens on host; under CUDA-graph replay the param is baked in, so descriptor cost is not on the timed path as long as pointers don't change. (inference; see [[cuda-graphs]])

## Further reading (same author, relevant to Hopper kernels)
- [GPU Memory Consistency Model](https://yang-yifan.github.io/blogs/memory_model/memory_model.html) – generic vs async proxy, `fence.proxy.async`, mbarrier acquire/release, TMA→CUDA core and CUDA core→TMA patterns. Digested into [[mbarrier]].
- [Tensor Core MMA Swizzle Layout](https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html) – Sec. 9 "Compatibility with TMA": TMA box inner width must equal exactly one swizzle-atom width; see [[swizzling]], [[tma]].
- [4 Ways to Do CuTe Copy](https://yang-yifan.github.io/blogs/cute_copy/cute_copy.html) – non-TMA CuTe copies (ld.global/cp.async), TV layouts, TiledCopy.
- [Blackwell MMA SMEM Descriptor](https://yang-yifan.github.io/blogs/smem_descriptor/smem_descriptor.html) – 64-bit smem matrix descriptor for wgmma/tcgen05 (Blackwell focus); see [[wgmma]].
- [Using PDL](https://yang-yifan.github.io/blogs/pdl/pdl.html) → [[yangyifan-pdl]]; [CuTe Layout and Tensor](https://yang-yifan.github.io/blogs/cute_layout/cute_layout.html) → [[yangyifan-cute-layouts]]; [Compute-bound](https://yang-yifan.github.io/blogs/reg_tile/reg_tile.html) → [[yangyifan-register-tiling]].
- Author's own references: Colfax "Mastering the NVIDIA TMA" tutorial; Hot Chips 2020 (A100) / 2022 (H100) slides.

## Related
[[tma]] [[mbarrier]] [[cute-layouts]] [[pdl]] [[swizzling]] [[wgmma]] [[warp-specialization]] [[cuda-graphs]] [[yangyifan-register-tiling]]
