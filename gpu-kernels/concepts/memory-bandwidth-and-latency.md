---
title: Memory Bandwidth and Latency (Little's law, bytes in flight)
type: concept
tags: [gpu, hopper, bandwidth, latency, littles-law, ilp, occupancy, decode]
source_url:
  - https://shreyansh26.github.io/post/2025-03-23_gtc25-maximize-memory-bandwidth-part-1
  - https://modal.com/gpu-glossary/perf/littles-law
  - https://modal.com/gpu-glossary/perf/latency-hiding
  - https://modal.com/gpu-glossary/perf/occupancy
  - https://www.nvidia.com/en-us/data-center/h200/
  - https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
---
# Memory Bandwidth and Latency (Little's law, bytes in flight)

**TL;DR**
- **Little's law**: bytes_in_flight = bandwidth × latency. To hit peak HBM you must keep that many bytes outstanding *at all times* across the chip (GTC25 S72683; Modal).
- GTC25: Hopper needs **> 32 KiB in flight per SM**; Blackwell > 40 KiB.
- Per-SM in-flight bytes ≈ bytes/load × loads/thread × threads/block × blocks/SM. Raise it with **vector width** (16 B loads), **ILP** (unrolled independent loads), **occupancy** (more warps), or **async copies** (TMA/cp.async – bytes in flight without registers).
- Occupancy and ILP are interchangeable sources of concurrency; registers are the shared budget. Hopper GEMMs often run at single-digit occupancy (Modal) using deep async pipelines instead.
- For µs-scale kernels, latency (ramp-up, first-byte, tail) is a large fraction of runtime – the steady-state roofline is not reachable.

## Key ideas
- Modal example: 400-cycle latency × 1 op/cycle → 400 concurrent ops; latency-hiding example: 416-cycle dependent chain → 13 warps.
- Volkov (via Modal): ~30 warps to hide memory latency vs ~24 for arithmetic on older GPUs, because memory BW is lower than arithmetic BW.
- GTC25 levers, in order: (1) if registers available, unroll + vectorize; (2) if registers scarce, async copy to smem; (3) chunk > 2 KiB → TMA, < 1 KiB → LDGSTS, 1–2 KiB → benchmark.
- Vector loads: 64/128-bit (`float2`/`float4`/`uint4`), require alignment; Hopper L1/L2 line 128 B. One warp of LDG.128 = 512 B = 4 full lines.
- Async copies move bytes in flight out of the register file → more in flight per thread without spills, and lower MIO/LSU instruction pressure.
- Occupancy beyond what latency hiding needs can hurt (fewer registers/thread) (Modal).

## Numbers worth remembering
- Hopper: > 32 KiB in flight/SM; Blackwell: > 40 KiB/SM (GTC25 via notes).
- H200 SXM: 4.8 TB/s HBM3e, 141 GB (NVIDIA product page). H100 SXM: 3.35 TB/s (Modal) / "up to 3 TB/s" (Hopper tuning guide wording).
- H100: 132 SMs (Modal L1 page), 64 warps/SM, 65,536 regs/SM, 255 regs/thread max, 228 KB smem/SM (tuning guide; Modal).
- Cache line 128 B (GTC25).

## Derived estimates (all inference)
- Implied latency from GTC numbers: 32 KiB × 132 SMs ≈ 4.3 MB; 4.3 MB / 3.35 TB/s ≈ 1.3 µs. Real loaded latency is likely lower; treat "32 KiB" as a safe target that includes headroom.
- H200 with the same SM count and similar latency: 32 KiB × 4.8/3.35 ≈ **46 KiB/SM**, ≈ 6 MB chip-wide.
- Ways to hold 46 KiB/SM:
  - LDG.128 into registers: 46 KiB / 16 B ≈ 2900 loads → e.g. 8 warps × 32 lanes × 12 float4 (48 regs of payload per thread), or 16 warps × 6. Needs the compiler to issue all loads before any use.
  - TMA ring: e.g. 4 stages × 12 KiB or 6 stages × 8 KiB of W per SM, costs smem not registers; 1–2 producer threads.
- Per-SM bandwidth share on H200: 4.8 TB/s / 132 ≈ 36 GB/s ≈ 36 KB/µs. A 2 µs kernel therefore moves only ~70 KB per SM in total – barely 1.5× the in-flight requirement. So for the smallest shapes the kernel is mostly "issue everything at t=0 and wait one latency"; launch/prologue overhead and the tail dominate.
- Floors 2–17 µs at 4.8 TB/s ↔ ~9.6–82 MB of weights (inference from the problem statement).

## Code patterns
```cpp
// Weight-streaming GEMV-style inner loop: U independent 16B loads per thread
const uint4* w4 = reinterpret_cast<const uint4*>(W + row * K);  // 8 BF16 per uint4
uint4 buf[U];
#pragma unroll
for (int u = 0; u < U; ++u)
  buf[u] = __ldg(&w4[lane + u * 32]);   // or ld.global.nc.L1::no_allocate; all U in flight
#pragma unroll
for (int u = 0; u < U; ++u)
  acc = dot8_bf16(buf[u], x_smem + (lane + u * 32) * 8, acc); // consume after issue
```
- `#pragma unroll` + separate load/consume loops forces ILP; verify in SASS (`cuobjdump -sass`) that LDG.E.128s are contiguous.
- Streaming data read once: cache hints (`ld.global.nc`, `L1::no_allocate`, `evict_first`) avoid polluting L1/L2 (inference; behavior not covered by these sources).

## Pitfalls
- Grid too small: if only a fraction of SMs get work, chip-wide bytes in flight scale down proportionally. Use ≥ 1 CTA per SM (132) and split K if n/BN < 132 (inference).
- Dependent loads (pointer chasing, loop-carried address math the compiler won't hoist) cap ILP at 1.
- Register spills from excessive unroll turn loads into local-memory traffic.
- Tail effect: last wave of CTAs leaves SMs idle; for µs-scale kernels prefer exactly one balanced wave or persistent CTAs (inference).
- Measuring with warm L2 overstates bandwidth; the judge flushes L2 (problem statement).
- **Measured (H100, [[findings-log]] 2026-11-09):** a register-held W stream reached only ~2.0 TB/s (ncu base clocks) at ~2 warps per scheduler. An async-copy SMEM ring reached 2.1–2.5 TB/s at **1 warp per scheduler**.
- If one operand (e.g. x) is still loaded synchronously from L2 inside the loop, the ring doesn't help. A hand-written ring ran at 0.71 TB/s until x went through it too.

## Relevance to the serving track
- (inference) `muse_bf16_proj` is the textbook Little's-law kernel: m ≤ 256, AI ≈ m FLOP/B for m ≪ n,k, so for m ≤ ~128 the time is bytes / (achieved BW). Achieved BW is set by whether every SM has ≥ ~46 KiB outstanding from the first cycle.
- (inference) Design rules: 1 wave of CTAs covering all 132 SMs; each CTA issues its first ~46 KiB of W immediately (TMA multi-stage or unrolled LDG.128); x loaded once into smem (tiny); split-K with a cheap reduction when n is too small to fill the chip; [[pdl]] to hide the prologue behind the previous graph node; no L2 reuse to count on.
- (inference) For m=128/256 the per-byte compute is no longer negligible (AI ≈ 100–230 FLOP/B, near H200's ridge ≈ 206) → use tensor cores ([[wgmma]]) so compute stays hidden under the memory stream.

## Related
[[gpu-memory-hierarchy]] · [[roofline-and-arithmetic-intensity]] · [[gtc25-s72683-memory]] · [[modal-gpu-glossary]] · [[tma]] · [[pdl]] · [[cuda-graphs]] · [[wgmma]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[cp-async]] · [[warp-specialization]] · [[benchmarking-kernels]]
