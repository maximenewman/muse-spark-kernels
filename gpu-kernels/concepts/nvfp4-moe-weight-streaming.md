---
title: "NVFP4 (W4A16) MoE experts with in-register dequant and mma.sync"
type: concept
tags: [gpu, moe, nvfp4, fp4, w4a16, dequant, mma, tma, cp-async-bulk, mbarrier, muse]
source_url:
  - ../muse_moe_w4a16/submission.py
  - ../muse_moe_w4a16/TASK.md
---
# NVFP4 MoE experts: in-register dequant + mma.sync

**TL;DR**
- `muse_moe_w4a16`: top-8 of 256 experts, H = I = 4096, NVFP4 weights (E2M1 + E4M3 scale per 16 + FP32 global), BF16 activations.
- Four raw-CUDA kernels (route → gate/up GEMM + SiLU → down GEMM + norm partials → norm + weighted sum) beat SGLang's Marlin path by **1.60× on H100**.
- **The kernels are instruction-bound, not bandwidth-bound.** Dequantizing FP4 costs about 20 instructions per thread per 16-element k-step. So occupancy and instruction count decide speed, not bytes in flight.

## Key ideas
- **Fragment-order packing in `prepare()`:** one u32 holds one thread's 8 FP4 values for one `m16n8k16` A fragment (rows g / g+8, k 2t.., 2t+8..). A warp's 128 B per k-step is a single coalesced chunk. Each CTA's per-stage chunk is contiguous, so one `cp.async.bulk` moves it.
- **FP4 → BF16 with a shift and a mask:** put the 3 magnitude bits at BF16 bits 6–8 and the sign at bit 15. The result reads as e2m1 × 2^-126; the 0.5 code becomes a BF16 subnormal, which `HMUL2.BF16` handles. The bits are laid out so that three registers need only `(q << s) & 0x81C081C0` (s = 0, 3, 6) and the fourth needs 6 ops: about 11 integer ops per 8 weights, down from about 15 with Marlin-style nibbles.
- **E4M3 scale → BF16 × 2^119 with `prmt` + one IMAD:** replicate the byte into both halves, then `* 16 + 0x77807780`. This is exact for normal, positive scales; `prepare()` checks that and falls back otherwise. Each weight is then `HMUL2(fp4bits, scale)`, exactly representable in BF16. The remaining 2^7 goes into the FP32 epilogue together with the global scale.
- **Gate and up in one A fragment:** rows 0–7 are gate rows i, rows 8–15 the matching up rows. So c0/c2 of the accumulator are gate/up for the same (token, i), and SiLU·up happens in registers.
- **Routing without host sync:** a one-CTA route kernel writes expert-grouped sorted positions and work items (expert, first position, up to TB tokens) plus their count. GEMM grids are (row tiles, slots), and each CTA loops `item += slots`.

## Numbers (H100, cold L2, graph replay)
| load | SGLang Marlin µs | ours µs | speedup | floor at 3.35 TB/s |
|---|---|---|---|---|
| decode 1 | 85.6 | 38.4 | 2.23× | ~20 µs (2.25 experts) |
| decode 32 | 180.0 | 143.2 | 1.26× | ~120 µs (14 experts) |
| verify 32 | 642.9 | 430.8 | 1.49× | ~205 µs (24 experts) |
| geomean of 6 loads | | | **1.60×** | ideal ≈ 2.4× |

## Pitfalls (each measured)
- **Occupancy beats pipeline depth.** A 2–4-stage ring with 4 CTAs per SM beat 6–8 stages with 1–2. Larger token blocks (TB 32/64) cost shared memory and lost, even though they cut re-dequantization.
- **Split-K didn't help at batch 1.** The extra warps just stalled on data. A copy-only variant still took 17 µs for 38 MB.
- **Predicated `HMMA` on a runtime token count** adds a `WARPSYNC` per `mma`, plus address recomputation. Making the first token block unconditional gave 2–4%.
- **`torch.cuda._compile_kernel` runs `ctypes.util.find_library` on every launch,** which starts `gcc`/`ld` subprocesses. Use your own `ctypes` NVRTC + `cuLaunchKernel` wrapper instead (judge rule: no subprocesses).
- **Tuning-script bug:** a re-timing generator that never applied the config it claimed to time made every config look identical. Always set state inside the loop that measures it.

## Next
- `wgmma` with a dequantize-to-shared-memory-once stage per (expert, row tile) for verify: removes the per-item re-dequant on hot experts (up to 161 tokens).
- L2 prefetch of the down-projection weights during the gate/up GEMM for decode 1.

## Related
[[weight-only-quantization]] · [[mma-ptx]] · [[mbarrier]] · [[tma]] · [[gemv-and-skinny-gemm]] · [[register-direct-weight-streaming]] · [[split-k-and-stream-k]] · [[findings-log]] · [[rorqual-h100-workflow]]
