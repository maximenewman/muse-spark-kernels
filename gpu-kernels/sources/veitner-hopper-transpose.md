---
title: Veitner – Making Matrix Transpose Really Fast on Hopper GPUs
type: source
tags: [gpu, hopper, tma, swizzle, transpose, bandwidth, smem]
source_url:
  - https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/
  - https://github.com/simveit/effective_transpose
---
# Veitner – Making Matrix Transpose Really Fast on Hopper GPUs

**TL;DR**
- 32768×32768 FP32 transpose on Hopper using raw CUDA + TMA (`cuTensorMapEncodeTiled`, `cp.async.bulk.tensor`), no CUTLASS.
- Three steps: naive TMA + smem transpose (875 GB/s, 26.5%) → `CU_TENSOR_MAP_SWIZZLE_128B` + swizzle-aware indexing (1252 GB/s, 37.9%) → more work per thread ("batching", 2771 GB/s, 84.0%).
- The biggest win was NOT swizzling but giving each thread several independent elements (fewer threads, more work/ILP per thread). Swizzle removes smem bank conflicts; batching removes per-thread overhead and raises useful work per issued instruction.
- "% of max" is computed against 3300 GB/s hard-coded in the repo (`bandwidth / 3300 * 100`), i.e. ~H100 SXM HBM3 peak.
- Swizzle index formula is taken from a GTC 2024 talk, not official docs; see [[swizzling]] for the official Table-27 form.

## Key ideas
- **Structure of each kernel**: one thread issues a 2D TMA load of a 32×32 FP32 tile (4 KB) into smem (aligned 1024 B), all threads wait on a `cuda::barrier` that tracks transaction bytes, threads transpose smem→smem into a second 4 KB buffer, `fence_proxy_async_shared_cta()` makes generic-proxy smem writes visible to the async (TMA) proxy, then one thread issues a TMA store and waits for it to finish reading smem.
- **Naive smem transpose** `smem_tr[col*B + row] = smem[row*B + col]` with B=32 floats: writes stride 32 floats = 128 B → all lanes of a warp hit the same bank → 32-way conflict (see [[smem-bank-conflicts]]).
- **128B swizzle**: TMA writes the tile into smem with the 16-byte chunks of each 128 B row XOR-permuted by (row mod 8). Reading/writing with the matching swizzle function makes both the row read and the column write conflict-free. TMA store with the same swizzle mode un-permutes on the way out.
- **Batching**: block = (32·32)/BATCH_SIZE threads; each thread handles BATCH_SIZE consecutive columns of a row in an unrolled loop. Repo ships BLOCK_SIZE=32, BATCH_SIZE=8 → 128 threads/CTA; the post also suggests BLOCK_SIZE=64, BATCH_SIZE=16.
- Grid = 1024×1024 CTAs of tiny 4 KB tiles; each CTA does exactly one load and one store (no pipelining/multi-stage) – throughput comes from many resident CTAs per SM providing bytes in flight.

## Numbers worth remembering
(from post + repo README; GPU only described as "Hopper", peak assumed 3300 GB/s in code)
| Kernel | Time | BW | % of 3300 GB/s |
|---|---|---|---|
| naive (TMA, unswizzled) | 9.81 ms | 875.46 GB/s | 26.5% |
| 128B swizzle | 6.86 ms | 1251.76 GB/s | 37.9% |
| swizzle + batch | 3.10 ms | 2771.35 GB/s | 84.0% |
Bytes counted = 2 × 4 GiB (read + write).

## Code patterns
Swizzle helpers (repo `transpose_swizzle_batched.cu`, FP32, B = BLOCK_SIZE = 32 floats = 128 B rows):
```cpp
template <int BLOCK_SIZE>
__device__ int calculate_col_swizzle(int row, int col) {
  int i16 = (row * BLOCK_SIZE + col) * 4 >> 4; // linear byte offset / 16 -> index of 16B chunk
  int y16 = i16 >> 3;                          // which 128B row (8 chunks per row)
  int x16 = i16 & 7;                           // chunk within the row
  int x16_swz = y16 ^ x16;                     // XOR chunk index with row -> Swizzle<3,4,3>
  return ((x16_swz * 4) & (BLOCK_SIZE - 1)) + (col & 3); // back to float col; keep pos inside chunk
}
```
`calculate_row_swizzle` is the same with (row,col) swapped for the transposed buffer.

TMA load / barrier / store skeleton:
```cpp
__shared__ alignas(1024) float smem_buffer[B*B];        // 1024B align = 128B-swizzle period
if (threadIdx.x == 0) { init(&bar, blockDim.x); cde::fence_proxy_async_shared_cta(); }
__syncthreads();
if (threadIdx.x == 0) {
  cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
  token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer)); // expect-tx bytes
} else token = bar.arrive();
bar.wait(std::move(token));
// ... smem transpose with swizzled indices ...
cde::fence_proxy_async_shared_cta();   // generic smem writes -> visible to TMA (async proxy)
__syncthreads();
if (threadIdx.x == 0) {
  cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_tr, y, x, &smem_buffer_tr);
  cde::cp_async_bulk_commit_group();
  cde::cp_async_bulk_wait_group_read<0>(); // smem must not be freed before TMA has read it
}
```
Tensor map: `cuTensorMapEncodeTiled(..., CU_TENSOR_MAP_DATA_TYPE_FLOAT32, rank=2, ptr, size={W,H}, stride={W*4}, box={32,32}, elem_stride={1,1}, INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, L2_PROMOTION_NONE, OOB_FILL_NONE)`.

## Pitfalls
- Forgetting `fence_proxy_async_shared_cta()` between generic smem writes and a TMA store = stale data (async proxy has no same-address ordering with generic proxy; see [[gtc25-s72683-memory]]).
- Swizzled smem buffers must be aligned to the swizzle repeat period (1024 B for 128B mode) or the address-based XOR phase is off (official docs: [[swizzling]]).
- 128B swizzle requires box inner dimension ≤ 128 B (32 FP32 / 64 BF16).
- One-tile-per-CTA with no multi-stage pipeline only works because there are huge numbers of CTAs; for a small grid you need a multi-stage pipeline to keep bytes in flight.

## Relevance to the serving track
- (inference) `muse_bf16_proj` has no transpose, but the same lesson applies: the jump from 38% → 84% came from more work per thread/CTA, not cleverer layout. For weight streaming, the analogous lever is more outstanding bytes per SM (deeper TMA pipeline or more unrolled 16-B loads), see [[memory-bandwidth-and-latency]].
- (inference) If W tiles are TMA-loaded into smem for [[wgmma]], use 128B swizzle with 64-element BF16 inner boxes; that is the layout wgmma expects (see [[swizzling]], [[tma]]).
- (inference) 84% of peak on a 8 GiB problem is a steady-state number; the judge's 2–17 µs shapes are dominated by ramp-up/latency, so this ceiling is optimistic for us.

## Related
[[swizzling]] · [[smem-bank-conflicts]] · [[tma]] · [[memory-bandwidth-and-latency]] · [[cute-layouts]] · [[feldmann-smem-microbenchmarks]] · [[mbarrier]]
