---
title: GTC25 S72683 – CUDA Techniques to Maximize Memory Bandwidth and Hide Latency
type: source
tags: [gpu, hopper, bandwidth, latency, littles-law, tma, ldgsts, memory-model, gtc]
source_url:
  - https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/
  - https://www.nvidia.cn/on-demand/session/gtc25-s72683
  - https://shreyansh26.github.io/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-1
  - https://shreyansh26.github.io/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-2
  - https://forums.developer.nvidia.com/t/can-you-post-a-pdf-of-cuda-techniques-to-maximize-memory-bandwidth-and-hide-latency/358985
---
# GTC25 S72683 – CUDA Techniques to Maximize Memory Bandwidth and Hide Latency

**Read status:** the NVIDIA On-Demand page only yielded the title (video page, no abstract text extracted). Presenters per search snippet: Athena Elafrou and Allard Hendriksen (NVIDIA DevTech). No official PDF exists (NVIDIA forum thread asks for one). Content below is from Shreyansh Singh's two-part notes; code on slides was images, so snippets here are reconstructions of the named APIs, not verbatim slide code.

**TL;DR**
- Little's law for memory: **bytes-in-flight = bandwidth × mean latency**. Hardware fixes bandwidth & latency; software controls bytes in flight.
- **Hopper needs > 32 KiB in flight per SM** to saturate DRAM; Blackwell > 40 KiB. Bandwidth grows faster than SM count, so the per-SM requirement keeps rising.
- Bytes in flight/SM ≈ bytes/load × loads/thread × threads/block × blocks/SM; **loads per thread** is the main knob (ILP via unrolling, DLP via 64/128-bit vector loads).
- If registers allow → unroll + vectorize. If not → async copies to smem (LDGSTS `cp.async` or TMA). Chunk size: **> 2 KiB → TMA, < 1 KiB → LDGSTS, 1–2 KiB → benchmark**.
- Part 2: memory model – scopes, release/acquire vs relaxed, async proxy fences (generic stores before TMA reads need `fence.proxy.async`), cheap cluster barrier init.

## Key ideas
### Part 1 – maximizing memory throughput
- **ILP**: more independent loads per thread, mostly by unrolling (sometimes manual).
- **DLP**: vector loads (`float2`/`float4`, 64/128-bit); need alignment; Hopper L1 and L2 line = 128 B.
- **Register cost**: in-flight bytes must land in registers → ILP/DLP raise register use, risk spills. Low-compute kernels usually OK.
- **Async copies** (global→smem, bypass registers): free registers, less L1 traffic, fewer instructions (less MIO pressure), overlap via prefetch distance n or producer/consumer. Best for iterative `GMEM→SMEM→compute→GMEM` kernels and low-occupancy, compute-heavy kernels.
- **LDGSTS** (Ampere+): async `smem[i]=gmem[j]` of 4/8/16 B. 16 B size+alignment → *L1 bypass* mode (compiler prefers it); 4/8 B → L1 access mode. Sync with `__pipeline_commit()` / `__pipeline_wait_prior(n)` (wait all but last n). Multi-stage: `NUM_STAGES`, always wait on the oldest stage; loads/thread = 2 × stages in the talk's example. Issue in converged code; sync before consumers since different threads fetch.
- **TMA** (Hopper+): 1D bulk (`UBLKCP`, contiguous) and 2D+ tensor copies. Warp-uniform: issue from **one thread**; otherwise compiler emits a peeling loop. Use `cooperative_groups::invoke_one` to avoid the peel. TMA 1D: src/dst 16 B aligned, size multiple of 16 B. APIs: `cuda::memcpy_async()` + `cuda::barrier` (has fallback for unaligned), `cuda::device::memcpy_async_tx()` (UB if not 16 B-multiple), `cuda::ptx` for fine control (one thread does arrive + expect-tx; waiting only polls arrival – cheaper than `barrier.wait()`). TMA prefetch pipelines need one smem barrier per stage (the `cuda::pipeline` construct can't be used). `thrust::transform` (CCCL) uses TMA and auto-tunes bytes-in-flight.

### Part 2 – memory model and hiding latency
- Scopes: thread / block (`.cta`) / cluster / device (`.gpu`) / system (`.sys`). Block-scope release/acquire is cheap (L1); cluster/device require L2 flush on release and L1 invalidate on acquire.
- Relaxed = single coherent load/store, no ordering (one value). Release-acquire orders surrounding accesses (multiple values).
- **Async proxy** (TMA, tensor cores, …) has no same-address ordering with the generic proxy: a generic smem store can be overtaken by a TMA read of the same smem → stale data stored to global. Fix: async proxy fence. Waiting on an mbarrier after a bulk async copy gives implicit ordering.
- `st.async`/`red.async` (Hopper) write remote DSMEM and signal its mbarrier.
- Cluster sync: fast init = relaxed arrive + `fence_mbarrier_init` release fence instead of a full cluster sync; only worth it for short kernels.

## Numbers worth remembering
- Hopper: > 32 KiB in flight per SM to saturate DRAM; Blackwell: > 40 KiB (talk, via notes).
- Hopper L1/L2 cache line: 128 B.
- LDGSTS sizes 4/8/16 B; L1-bypass requires 16 B.
- TMA 1D: 16 B alignment, size multiple of 16 B.
- TMA vs LDGSTS crossover: 1–2 KiB per chunk.

## Code patterns
(Reconstructed from the API names in the notes; check against [[tma]].)
```cpp
// ILP + DLP: each thread keeps U independent 16-B loads outstanding
float4 r[U];
#pragma unroll
for (int u = 0; u < U; ++u) r[u] = reinterpret_cast<const float4*>(src)[base + u*stride]; // all issued before first use
#pragma unroll
for (int u = 0; u < U; ++u) acc += dot(r[u]);  // consume after all loads in flight
```
```cpp
// LDGSTS multi-stage prefetch (Ampere-style pipeline primitives)
__pipeline_memcpy_async(&smem[stage][tid], &gmem[idx], 16); // 16 B -> L1-bypass path
__pipeline_commit();                                        // close this stage's batch
__pipeline_wait_prior(NUM_STAGES - 1);                      // wait for the oldest stage only
__syncthreads();                                            // other threads fetched what I consume
```

## Pitfalls
- Unrolling that the compiler serializes (loads interleaved with dependent uses) gives no ILP – check SASS that all LDGs issue before the first FMA.
- Issuing TMA from a whole warp without `invoke_one`/`elect` → serialized peeling loop.
- `memcpy_async_tx` with non-16-B-multiple sizes is UB.
- Missing async proxy fence between generic smem writes and TMA store.

## Relevance to the serving track
- (inference) Biggest single number for `muse_bf16_proj`: > 32 KiB in flight per SM on H100. H200 has 4.8 TB/s vs H100's 3.35 TB/s with the same 132 SMs, so if latency is similar it needs ~1.4× more, i.e. roughly ≥ 45 KiB/SM. Over 132 SMs that is ~6 MB outstanding – comparable to a whole small weight matrix, meaning for the 2 µs shapes the kernel should essentially issue (almost) all its loads immediately.
- (inference) Options to reach it: (a) LDG.128 with deep unroll: 45 KiB/SM ÷ 16 B ≈ 2900 loads in flight/SM, e.g. 1024 threads × ~3 float4, or 256 threads × ~12 float4; (b) TMA bulk/tensor loads into a multi-stage smem ring (e.g. 4–8 stages × 8–16 KiB), which costs no registers – the talk's guidance (chunks > 2 KiB → TMA) points to TMA for weight tiles.
- (inference) Each CTA's x (m×k BF16, ≤ 256×k) is small; weight W is the stream. Cold L2 means every W byte comes from HBM, so there is no reuse to exploit – only concurrency.
- (inference) Use [[pdl]] to overlap the prologue (barrier init, tensor-map prefetch) with the previous kernel's tail inside the [[cuda-graphs]] replay.

## Related
[[memory-bandwidth-and-latency]] · [[tma]] · [[gpu-memory-hierarchy]] · [[pdl]] · [[cuda-graphs]] · [[veitner-hopper-transpose]] · [[modal-gpu-glossary]] · [[cp-async]] · [[mbarrier]] · [[thread-block-clusters]]
