---
title: Thread Block Clusters, DSMEM, and TMA Multicast
type: concept
tags: [gpu, hopper, blackwell, clusters, dsmem, tma, multicast, mbarrier]
source_url:
  - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
  - https://github.com/pranjalssh/fast.cu  (h100/matmul/matmul_8.cuh)
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
  - https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
---
# Thread Block Clusters (CGA), DSMEM, TMA Multicast

**TL;DR**
- Hopper adds a level between CTA and grid: a **cluster** of CTAs co-scheduled on SMs of one GPC. Their SMEM forms **distributed shared memory** (DSMEM), addressable via `shared::cluster` / `mapa`. [semianalysis]
- **TMA multicast** loads one GMEM tile once and writes it into the SMEM of several CTAs in the cluster (selected by a `ctaMask`), cutting L2/HBM traffic. [semianalysis, worklog]
- Worklog: cluster (2,1,1) + multicast of the shared operand: 704 → **734 TFLOPs**. Vertical pairs were best; larger clusters were slower. [worklog]
- Consumers must release a stage to **every** CTA that multicast into it, so cluster-scope mbarrier arrives go through `mapa`. [fast.cu]
- Blackwell CTA pairs (2-SM MMA) build on the same mechanism. [semianalysis, colfax]

## Key ideas
- **Declare:** `__cluster_dims__(X,Y,Z)` on the kernel (or the cluster launch attribute). Rank via `%cluster_ctarank`, cluster id via `%clusterid`. [worklog, fast.cu]
- **Sync:** `barrier.cluster.arrive` + `barrier.cluster.wait` after mbarrier init, so every CTA's barriers exist before peers signal them. [fast.cu]
- **Multicast semantics (fast.cu kernel 8, CLUSTER_M=2):** each CTA computes a different M-block but the same N-block, so B is shared. Only `rank_m == 0` issues the B load with `.multicast::cluster` and a mask of both CTAs. Each CTA still loads its own A. Each destination CTA's `full[q]` barrier receives `complete_tx` for the bytes landing in *its* SMEM, so every CTA's `expect_tx` stays the full A+B bytes.
- **Release across the cluster:** a stage in CTA r can be refilled only when consumers in **all** CTAs are done with it, because the multicast write targets all of them. So `empty[q]` is initialized with count `num_consumers * CLUSTERS`, and each consumer arrives on `empty[q]` of every CTA via `mapa.shared::cluster` + `mbarrier.arrive.shared::cluster`. The producer waits with `try_wait.parity.acquire.cluster`.
- **Blackwell:** CTA pairs (ranks differing in the last bit → same TPC) let one `cta_group::2` MMA span two SMs. B is split and exchanged over DSMEM. [semianalysis] In CuTe the atom's thread mode becomes the peer-CTA mode. [colfax]

## Numbers worth remembering
- Cluster of 2 on H100 at 4096³: +4% (704 → 734). Clusters > 2 were slower. [worklog]
- With clusters, the scheduler runs over `NUM_SM / CLUSTERS` = 64 cluster slots and 16/CLUSTER_M × 8/CLUSTER_N tile groups. [fast.cu]
- Portable max cluster size is 8 (16 with opt-in on H100). That's general CUDA knowledge, not from these sources.

## Code patterns (fast.cu matmul_8.cuh)
```cuda
__global__ __launch_bounds__(384) void __cluster_dims__(2, 1, 1) kernel(...);

uint32_t rank; asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank));
// after mbarrier.init by thread 0:
asm volatile("barrier.cluster.arrive;\n");  asm volatile("barrier.cluster.wait;\n");

// multicast TMA: one issuer, data + complete_tx land in every CTA in cluster_mask
asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
             " [%0], [%1, {%3, %4, %5}], [%2], %6;"
             :: "r"(dst_smem), "l"(tma_map), "r"(mbar), "n"(0), "r"(row), "r"(col/64), "h"(cluster_mask) : "memory");

// arrive on the mbarrier at the same smem offset in CTA `cta_id` (DSMEM)
asm volatile("{\n\t.reg .b32 remAddr32;\n\t"
             "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"      // local smem addr -> peer's cluster addr
             "mbarrier.arrive.shared::cluster.b64 _, [remAddr32], %2;\n\t}"
             :: "r"(smem_addr), "r"(cta_id), "r"(count));
// consumer: if (tid < CLUSTERS) arrive_cluster(&empty[q], tid);   // one thread per peer CTA

// producer waits with cluster-scope acquire
"mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;"
```

## Pitfalls
- A cluster can deadlock if peers signal barriers before they're initialized. Do a cluster barrier after init. [fast.cu pattern]
- With multicast, a slow CTA stalls its peers (shared stage lifetime). Bigger clusters amplify this, which is consistent with the worklog's finding that >2 was slower.
- Grid dims must be divisible by the cluster dims. fast.cu asserts `(M/BM) % CLUSTER_M == 0`.
- (inference) Clusters constrain placement (all CTAs in one GPC). With 132 SMs over GPCs of uneven size, large clusters can leave SMs idle and lower the effective SM count.

## Relevance to the serving track
- (inference) For x·Wᵀ with small m, **W tiles are unique per CTA**, so multicasting W saves nothing. The shared operand is x ([m,k], ≤ 256×k bf16), which is L2-resident after the first touch even with a cold L2. Multicasting x saves a little L2 bandwidth but isn't the bottleneck.
- (inference) Where clusters *do* help small-m: **split-K reduction through DSMEM.** CTAs in a cluster each stream a K-slice of the same W rows, then reduce partial outputs via `st.shared::cluster`/`ld.shared::cluster` or `red` into the leader's SMEM. That avoids a second kernel or global atomics, which matters for router/pooler layers whose n/64 tile count < 132.
- (inference) The cluster launch attribute is capturable in [[cuda-graphs]]. Check that cluster scheduling doesn't add launch latency at 2–5 µs scale. Measure cluster=1 vs 2 vs 4.

## Related
[[tma]] · [[warp-specialization]] · [[persistent-kernels-and-tile-scheduling]] · [[tcgen05-tmem]] · [[wgmma]] · [[cuda-graphs]] · [[h100-gemm-worklog-cudaforfun]] · [[semianalysis-tensor-core-evolution]]
