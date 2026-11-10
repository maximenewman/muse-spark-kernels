---
title: GPU Memory Hierarchy (Hopper H100 / H200)
type: concept
tags: [gpu, hopper, h100, h200, memory-hierarchy, registers, smem, l1, l2, hbm]
source_url:
  - https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
  - https://www.nvidia.com/en-us/data-center/h200/
  - https://modal.com/gpu-glossary/device-software/memory-hierarchy
  - https://modal.com/gpu-glossary/device-hardware/l1-data-cache
  - https://modal.com/gpu-glossary/perf/occupancy
  - https://modal.com/gpu-glossary/perf/memory-bandwidth
  - https://shreyansh26.github.io/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-1
---
# GPU Memory Hierarchy (Hopper H100 / H200)

**TL;DR**
- Registers (per thread) → shared memory / L1 (per SM, same physical SRAM) → L2 (chip-wide) → HBM (GPU RAM). Distributed shared memory (DSMEM) links smem across a thread-block cluster on Hopper.
- H100 per SM: 64K × 32-bit registers, 256 KB combined L1/tex/smem, up to 228 KB smem (227 KB per block). L2 50 MB. HBM3 80 GB (tuning guide).
- H200 SXM: 141 GB HBM3e at 4.8 TB/s (NVIDIA). Same Hopper SM; the H200 page fetched gives no L2/SM-count numbers.
- Cache line 128 B for L1 and L2 on Hopper (GTC25).
- For weight streaming with cold L2, only the HBM link matters for W; everything else is staging.

## Key ideas
- **Registers**: fastest (~10× L1 per Modal); private to a thread except tensor-core fragments; register pressure lowers occupancy. Max 255 regs/thread.
- **Shared memory**: programmer-managed scratchpad carved from L1 SRAM; 32 banks × 4 B ([[smem-bank-conflicts]]). Carveouts per SM: 0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB. Static smem ≤ 48 KB; above that requires dynamic smem opt-in (`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`). 1 KB reserved per block.
- **L1 data cache**: 256 KiB/SM including smem (Modal: ×132 SMs = 33 MiB on H100 SXM5). Serves LDG unless bypassed (LDGSTS 16-B L1-bypass mode; TMA writes smem directly).
- **L2**: 50 MB on H100 (A100: 40 MB). Coherency point for cluster/device scope (GTC25 part 2). L2-to-SM bandwidth increased vs Ampere (no figure given).
- **HBM**: H100 "up to 3 TB/s" (tuning guide) / 3.35 TB/s (Modal table, H100 SXM HBM3). H200 4.8 TB/s.
- **DSMEM**: blocks in a cluster can access each other's smem; portable cluster size 8, non-portable 16; access aligned to 32-B segments where possible; DSMEM bandwidth adds to L2 bandwidth (tuning guide).
- **TMA**: moves 1D–5D tensors gmem↔smem and between SMs in a cluster ([[tma]]).

## Numbers worth remembering
| Level | H100 (source) | H200 (source) |
|---|---|---|
| Registers | 64K × 32-bit / SM; 255/thread (tuning guide) | same Hopper SM (inference) |
| Max warps / blocks per SM | 64 / 32 (tuning guide, Modal) | same (inference) |
| Smem | 228 KB/SM, 227 KB/block (tuning guide) | same (inference) |
| L1+smem | 256 KB/SM (tuning guide; Modal 256 KiB) | same (inference) |
| SMs | 132 on H100 SXM5 (Modal L1 page) | not stated in fetched sources |
| L2 | 50 MB (tuning guide) | not stated in fetched sources |
| HBM capacity | 80 GB (tuning guide) | 141 GB HBM3e (NVIDIA) |
| HBM BW | up to 3 TB/s (tuning guide); 3.35 TB/s (Modal) | 4.8 TB/s (NVIDIA) |
| Cache line | 128 B L1/L2 (GTC25) | — |
| NVLink | 900 GB/s bidirectional, 18 links (tuning guide) | — |

## Code patterns
```cpp
// Opt in to > 48 KB dynamic smem (needed for deep TMA rings)
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
kernel<<<grid, block, 227 * 1024, stream>>>(...);
extern __shared__ __align__(1024) uint8_t smem[];   // 1024 B align for 128B-swizzled TMA tiles
```

## Pitfalls
- Combined L1/smem: max smem carveout leaves little L1 for LDG traffic (relevant only for non-TMA loads that rely on L1).
- L2 (50 MB on H100) can hold a whole small weight matrix – benchmarking with warm L2 gives fantasy bandwidth. The judge flushes L2 before every replay.
- Spec bandwidths differ between NVIDIA documents (3 vs 3.35 TB/s for H100); quote the source.

## Relevance to the serving track
- (inference) W path: HBM → (TMA → smem ring → wgmma) or (LDG.128 → registers → FMA/mma). Both fine; the TMA path keeps bytes in flight without registers, the register path avoids smem round-trip latency for tiny m.
- (inference) x path: x (≤ 256 × k BF16) loaded once per CTA into smem and broadcast; small vs W.
- (inference) 227 KB smem/CTA allows ~14 × 16 KiB stages – far more than the ~46 KiB/SM needed for H200 bytes in flight, so a 1-CTA/SM persistent design is not capacity-limited.
- (inference) Weights that fit in 50 MB L2 would be L2-resident in production decode loops, but this judge measures cold-L2, so optimize for HBM.

## Related
[[memory-bandwidth-and-latency]] · [[roofline-and-arithmetic-intensity]] · [[smem-bank-conflicts]] · [[swizzling]] · [[tma]] · [[modal-gpu-glossary]] · [[gtc25-s72683-memory]] · [[thread-block-clusters]] · [[cp-async]] · [[gemv-and-skinny-gemm]]
