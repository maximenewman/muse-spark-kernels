---
title: "Split-K and Stream-K GEMM decomposition"
type: concept
tags: [gpu, gemm, split-k, stream-k, reduction, small-m, load-balancing]
source_url:
  - https://siboehm.com/articles/22/CUDA-MMM          # cuBLAS splitKreduce_kernel at 256^3
  - https://arxiv.org/abs/2301.03598                  # Stream-K (Osama et al.), abstract only
  - https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
---
# Split-K and Stream-K

**TL;DR**
- **Data-parallel GEMM** gives each CTA whole output tiles, so the CTA count is `ceil(M/BM)·ceil(N/BN)`. When that's below the SM count (small M, small N), most of the GPU idles.
- **Split-K** also partitions K into `S` slices. Each slice-CTA writes an f32 partial tile, and the partials are reduced (separate kernel, atomics, or last-CTA-reduces). CTAs become ×S.
  - Sourced: cuBLAS on A100 at 256³ runs `ampere_sgemm_64x32_sliced1x4_nn` (12.416 µs) + `splitKreduce_kernel` (6.912 µs) ([[siboehm-sgemm-worklog]]).
- **Stream-K** (Osama et al., 2023) divides the *total MAC-loop iterations* evenly across a persistent grid of CTAs. A CTA may cover a tail of one tile and a head of the next, and tiles are fixed up across CTAs. The abstract reports up to 14× and 6.7× peak speedups and more consistent performance across 32,824 GEMM shapes, with one tile config per precision.
- For `muse_bf16_proj` (m ≤ 256, N ∈ {256…8192}, K ∈ {1024…8192}), **split-K is the main lever for SM fill** (inference). Its costs are the f32 partial traffic and the reduction latency.

## Key ideas
### Split-K mechanics (inference except where cited)
- Grid: `(ceil(N/BN), ceil(M/BM), S)`. CTA z covers K range `[z·K/S, (z+1)·K/S)`.
- Partial size per CTA = BM×BN×4 B (f32). Total extra traffic = S·M·N·4 B written plus the same read back.
  - For small m that's tiny next to W. Example: m=8, N=2304, S=8 → 590 KB vs W = 37.7 MB.
  - It grows with m: m=256, N=8192, S=4 → 33.5 MB, comparable to W (16.8 MB for `o`). **So S should shrink as m grows.**
- **Reduction strategies:**
  1. **Second kernel** (cuBLAS `splitKreduce_kernel`): simplest, but one extra launch. In siboehm's example the reduce took 6.9 µs vs 12.4 µs for the GEMM.
  2. **`atomicAdd` f32 into the output/workspace:** needs a zeroed buffer (a memset node in the graph, or a zeroing pass) and an extra cast pass if the output is bf16. Order-nondeterministic.
  3. **Semaphore / "last CTA reduces"** (atomic ticket per output tile): each CTA writes its partial, `__threadfence()`, then `atomicAdd(&ticket[tile], 1)`. The CTA that sees `S-1` reads all partials, sums them in fixed order (deterministic), writes bf16, and resets the ticket to 0 for the next graph replay. That's what the user's kernel-track `submission.py` does ([[findings-log]]).
  4. **Intra-CTA split-K across warps:** warps of one CTA take different K sub-ranges and reduce through SMEM. No global traffic, but it doesn't add CTAs.
  5. **Cluster/DSMEM reduction on Hopper:** CTAs of a cluster reduce via distributed shared memory. Not covered by these sources; see [[tma]] and [[persistent-kernels-and-tile-scheduling]].
- **Determinism:** atomics give run-to-run bit differences. The challenge's correctness tolerance probably allows it, but ordered last-CTA reduction is safer (inference).

### Stream-K (paper abstract only)
- "Work-centric" rather than tile-centric. The aggregate inner-loop iterations are split evenly among the physical processing elements, so utilization doesn't depend on how the output tiling quantizes onto SMs.
- Claims: up to 14× and 6.7× peak speedup, consistent performance across 32,824 shapes, and a single tile config per precision instead of heuristics.
- Typical implementation (inference; from general knowledge of the CUTLASS stream-K design, not read here):
  - persistent grid = #SMs
  - each CTA handles `total_iters / #SMs` MAC iterations
  - partial tiles are fixed up through a workspace + flags
  - often hybridized: data-parallel for full waves, stream-K for the last partial wave

## Numbers worth remembering
- cuBLAS A100 256³ SGEMM: 12.416 µs GEMM + 6.912 µs `splitKreduce_kernel` (siboehm). **The reduce was ~36% of total time.** Fusing the reduction is worth it at µs scale (inference).
- Stream-K: 14× / 6.7× peak, 32,824 shapes (abstract).

## Code patterns
Last-CTA reduction (inference; the pattern behind option 3):
```cuda
// after computing f32 partial acc for (tile, split z)
store_partial(workspace + ((tile * S) + z) * BM * BN, acc);   // coalesced f32 stores
__threadfence();                                              // make partial visible device-wide
__syncthreads();
__shared__ int is_last;
if (threadIdx.x == 0) is_last = (atomicAdd(&ticket[tile], 1) == S - 1);
__syncthreads();
if (is_last) {
  float sum = 0; for (int s = 0; s < S; ++s) sum += load_partial(s);  // fixed order → deterministic
  write_bf16(out, sum);
  if (threadIdx.x == 0) ticket[tile] = 0;                     // self-reset for the next graph replay
}
```
- An optimization: the last CTA keeps its own `acc` in registers and adds only the other S−1 partials.

## Pitfalls
- Too large an S: K per CTA becomes a few BK tiles, so the pipeline prologue/epilogue dominates, and partial traffic grows ∝ S·m·N.
- Atomic f32 into a bf16 output isn't possible directly. You need an f32 workspace plus a cast.
- **Workspace zeroing under CUDA graphs:** memset nodes add latency. The self-resetting ticket avoids them.
- Wave quantization still applies to split-K (CTA count = tiles × S). Stream-K exists to remove it.

## Relevance to the serving track
- Dense projections at decode/verify (m ∈ {1, 8, 16, 32, 128, 256}) have few output tiles. E.g. N=1024 (unpooler slice) or N=2304 (qkv_gate) with BN=64–128 gives 8–36 CTAs on 132 SMs. **Split-K (or stream-K) is mandatory to reach HBM bandwidth** (inference).
- Choose S per (N, K, m) (inference):
  - Target ≈ 1–2 full waves.
  - Keep ≥ 2–4 BK iterations per CTA.
  - Reduce S as m grows, because partial traffic ∝ m.
- Stream-K's persistent-CTA scheme also suits attention split-KV at small batch → [[persistent-kernels-and-tile-scheduling]].

## Related
[[gemv-and-skinny-gemm]] · [[gemm-tiling-hierarchy]] · [[siboehm-sgemm-worklog]] · [[persistent-kernels-and-tile-scheduling]] · [[cuda-graphs]] · [[cp-async]] · [[findings-log]] · [[serving-track-playbook]]
