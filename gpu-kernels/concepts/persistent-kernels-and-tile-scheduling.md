---
title: Persistent Kernels and Tile Scheduling
type: concept
tags: [gpu, hopper, persistent-kernel, tile-scheduling, l2, rasterization, hilbert, wave-quantization, split-k]
source_url:
  - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
  - https://github.com/pranjalssh/fast.cu  (h100/matmul/matmul_8.cuh, matmul_11.cuh)
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
---
# Persistent Kernels and Tile Scheduling

**TL;DR**
- **Persistent kernel:** launch grid = #SMs (or #cluster slots). Each CTA loops over output tiles from a static or dynamic schedule. SMEM queue and barriers stay alive across tiles, so tile i's epilogue overlaps tile i+1's loads. [worklog]
- Persistence alone **regressed** the worklog from ~640 to 400 TFLOPs because of poor L2 locality. Grouped (16×8 tile-block) ordering fixed it: **660** at 83% L2 hit rate vs cuBLAS ~70%. [worklog]
- Hilbert-curve tile order added the last +6 (758 → 764). [worklog]
- 128 CTAs beat 132 (660 vs 655) because the tile groups divide evenly. Wave/group quantization is real. [worklog]
- SemiAnalysis: big MMA tiles worsen tile and wave quantization on small problems. [semianalysis]

## Key ideas
- **Why persistent:** amortize the prologue (barrier init, descriptor fetch, pipeline fill) once per CTA instead of per tile, and overlap the store-heavy epilogue (4572 cycles vs 1415 load / 703 MMA per block in the worklog) with the next tile's loads. [worklog]
- **L2 rasterization:** concurrent CTAs should work on nearby (m,n) tiles so the A row-panels and B column-panels they share hit in L2. fast.cu Schedule v1 assigns linear index `it*NUM_SM + block`, chunks it into TM×TN super-tiles (16×8 blocks), and walks inside each super-tile. [fast.cu]
- **Hilbert curve:** precompute the tile order on the host (`createHilbert(M_tiles, N_tiles, CORES, space)`) so consecutive tiles stay adjacent even across super-tile boundaries. The order is passed to the kernel as an int array. [fast.cu matmul_11]
- **Cluster-aware:** with clusters the scheduler runs over cluster slots (`NUM_SM/CLUSTERS`) and cluster-sized super-tiles; each CTA offsets by its rank. [fast.cu]
- **Quantization:** tiles ÷ SMs not integral → a partial last wave. SemiAnalysis notes both tile quantization and wave quantization hurt small matrices more as MMA atoms grow. [semianalysis]

## Numbers worth remembering
- Worklog at 4096³ (BM=128, BN=256 → 32×16 = 512 tiles): persistent naive 400, grouped 660 (128 SMs), 655 (132 SMs), Hilbert 764 (final). [worklog]
- L2 hit rate: 83% (grouped) vs ~70% (cuBLAS). [worklog]
- H100: 132 SMs, 50 MB L2 (behaves like 2 × 25 MB partitions per the author's footnote). [worklog]

## Code patterns
Static grouped schedule (fast.cu `Schedule<1,...>`):
```cuda
__device__ bool next(int& block_m, int& block_n) {
    int num = it * NUM_SM + block;                       // this CTA's it-th work item
    if (num >= total_blocks_m * total_blocks_n) return false;
    int cur_tile = num / (TM*TN), pos = num % (TM*TN);   // super-tile id, position inside
    block_m = TM * (cur_tile / (total_blocks_n/TN)) + pos / TN;
    block_n = TN * (cur_tile % (total_blocks_n/TN)) + pos % TN;
    ++it; return true;
}
// both producer and consumer warpgroups run identical schedule loops:
while (schedule.next(bm, bn)) { /* producer: TMA all k-blocks; consumer: MMA + epilogue */ }
```
Persistent launch (fast.cu): `kernel<<<NUM_SM /*=128*/, 384, smem>>>(...)`. Barrier phase `(qidx, p)` keeps running across tiles; it's not reset per tile.

Epilogue overlap with async TMA store (kernel 10): `cp.async.bulk.wait_group 0` before reusing `sC`, then write `sC`, then `bar.sync` consumers, then one thread `cp.async.bulk.tensor ... global.shared::cta` + `commit_group`. The next tile's MMAs proceed while the store drains.

## Pitfalls
- Persistence without locality-aware order can be **much slower** than a plain grid (−37% in the worklog).
- Using every SM isn't automatically better. Pick the CTA count so super-tiles divide evenly (128 > 132 here).
- Producer and consumer must agree on the exact same tile sequence. Any divergence deadlocks the mbarrier ring.
- Static schedules assume uniform tile cost. Variable-cost work (MoE experts with different token counts, causal attention) needs dynamic schedulers (atomic counter) or stream-K. That's inference; the sources don't cover it.

## Relevance to the serving track
- (inference) Cold L2 + weight streaming: each weight byte is read exactly once, so **L2 rasterization/Hilbert gains don't apply** to W. Ordering matters only to keep concurrent CTAs reading DRAM-friendly contiguous regions.
- (inference) **Wave quantization is the main lever.** Example: an n=4096 output with 64-row W tiles = 64 tiles on 132 SMs, so half the SMs idle and HBM is under-subscribed. Fix with split-K (each CTA streams a K-slice; reduce via DSMEM in a cluster, atomics, or a tiny second pass) or stream-K so #work-units ≈ multiple of 132 and every SM pulls bytes. For H200 also assume 132 SMs (H200 SM count isn't stated in these sources; verify with `cudaGetDeviceProperties`).
- (inference) At 2–17 µs a kernel is roughly 1–3 "tiles" per SM. A persistent loop buys little over a one-shot grid sized to the SM count. The prologue/fill latency matters more: overlap it with the previous kernel via [[pdl]] and keep the launch inside [[cuda-graphs]].
- (inference) Since the epilogue writes only m×n outputs (tiny), plain vectorized `st.global` beats staging through SMEM + TMA store at small m (fewer sync points).
- (inference) For NVFP4 MoE (other serving problems), tile cost varies per expert. Use a dynamic/grouped scheduler over (expert, n-tile, k-split) work items.

## Related
[[warp-specialization]] · [[thread-block-clusters]] · [[tma]] · [[wgmma]] · [[pdl]] · [[cuda-graphs]] · [[h100-gemm-worklog-cudaforfun]] · [[semianalysis-tensor-core-evolution]]
