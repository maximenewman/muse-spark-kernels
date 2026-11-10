---
title: Feldmann – Nvidia GPU Shared Memory Banks (microbenchmarks)
type: source
tags: [gpu, smem, bank-conflicts, microbenchmark, a100, vector-loads]
source_url: https://feldmann.nyc/blog/smem-microbenchmarks
---
# Feldmann – Nvidia GPU Shared Memory Banks (microbenchmarks)

**TL;DR**
- A100 microbenchmarks (Dec 2024) of `ld.volatile.shared` patterns; 1 CTA of 32×8 threads, 100k iterations.
- Model confirmed: 32 banks × 4 B/cycle, `bank(addr) = (addr/4) % 32`. Distinct addresses in the same bank serialize; same address = free broadcast; arbitrary multicast (many lanes, few addresses, no bank collision) is also free.
- Conflicts in different warps do **not** overlap: 8 warps each conflicting on their own bank are as slow as all on one bank.
- `ld.shared.v4` (16 B/lane) is conflict-free even though lanes' 16 B spans overlap banks – hardware staggers bank order; throughput stays 4 B/bank/cycle (so a v4 warp load takes ~4 bank-cycles = 4 "wavefronts").
- Vector multicast: pairs of lanes sharing a v2 → each lane effectively 2 loads/cycle; quads sharing a v4 don't scale further (suspected register-file write-port limit, untested).

## Key ideas
- Timing method: per-iteration loop of a single volatile shared load per lane; conflict-free baseline ≈ 1 load/bank/cycle at 1.4 GHz.
- Bank conflict cost is linear in the max number of distinct addresses per bank in a warp request (32-way → ~32×).
- What the post does NOT cover: ldmatrix, swizzle, latency, GB/s figures, Hopper.

## Numbers worth remembering
(A100, 1.4 GHz stated; times for 100k iterations, from the post)
| Pattern | Time |
|---|---|
| conflict-free lane i → bank i | ~0.57 ms (~1.4 G loads/bank/s) |
| all lanes bank 0 (32-way) | 18.2 ms (~32×) |
| each warp conflicts on a different bank | 18.2 ms (no cross-warp overlap) |
| full-warp broadcast | ~0.57 ms |
| random hash multicast | ~0.57 ms |
| `ld.shared.v4`, 512 B/iter | ~2.27 ms (4× data, 4× time) |
| `v2`, lane pairs share | ~0.57 ms |
| `v4`, lane quads share | ~1.14 ms |

## Code patterns
```cpp
float* ptr = &s[warp_id][lane_id];                       // lane i -> bank i
asm volatile("ld.volatile.shared.f32 %0, [%1];" : "=f"(r1) : "r"(addr)); // volatile: don't hoist/CSE
```
```cpp
unsigned int hash = (lane_id * 2654435761u) >> 16;        // Knuth hash -> pseudo-random index
volatile float* ptr = &sh[warp_id][hash % 32];            // multicast pattern, no 2 addrs share a bank w/ diff address
```
```cpp
float4* ptr = reinterpret_cast<float4*>(&sh[warp_id][lane_id * 4]);  // 16 B/lane
asm volatile("ld.volatile.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : ... : "r"(addr));
```

## Pitfalls
- Bandwidth ceiling per SM is 32 banks × 4 B = 128 B/cycle regardless of vector width; vectorizing smem loads reduces instruction count, not bytes/cycle.
- Benchmarks are A100; Hopper keeps the 32×4 B bank model (per [[modal-gpu-glossary]] bank-conflict page) but absolute numbers differ.
- Need `volatile`/inline PTX or the compiler eliminates repeated loads.

## Relevance to the serving track
- (inference) For a GEMV/skinny GEMM, x (m×k, tiny) is the natural smem-broadcast operand: all lanes reading the same x element is free (broadcast), so staging x in smem and streaming W from HBM through registers costs nothing in smem bandwidth.
- (inference) If W goes through smem (TMA → smem → ldmatrix/wgmma), smem bandwidth (128 B/clk/SM) is far above per-SM HBM share (~4.8 TB/s ÷ 132 SMs ≈ 36 GB/s/SM ≈ ~20 B/clk at ~1.8 GHz), so smem is not the bottleneck unless accesses are badly conflicted.
- (inference) Cross-warp conflicts serialize, so a single badly-laid-out warp can stall the SM's whole LSU path.

## Related
[[smem-bank-conflicts]] · [[swizzling]] · [[gpu-memory-hierarchy]] · [[veitner-hopper-transpose]] · [[mma-ptx]]
