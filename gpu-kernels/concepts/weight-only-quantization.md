---
title: Weight-only quantization (int8 / int4 / NVFP4) and in-kernel dequant
type: concept
tags: [gpu, quantization, int8, int4, nvfp4, mxfp4, dequant, gemv, decode, memory-bound]
source_url:
  - https://pytorch.org/blog/accelerating-generative-ai-2/
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
---
# Weight-only quantization and in-kernel dequant

**TL;DR**
- In decode, every matmul is a GEMV, so time ≈ weight bytes / HBM bandwidth. Halving bytes/param roughly halves time. gpt-fast: int8 weight-only took Llama-7B from 107.0 to 157.4 tok/s (+~50%). int4 g32 + GPTQ reached 202.1 tok/s ([[pytorch-gpt-fast-accelerating-genai-2]]).
- Only **weights** are quantized. Activations and compute stay bf16. Dequant happens in registers right after the load, inside the matmul kernel.
- **Fusion is mandatory.** Unfused int8 (separate dequant kernel) was *slower than fp16* in gpt-fast, because it re-materializes bf16 weights in HBM.
- Scale granularity trades accuracy for bytes: int8 used one scale per output row. int4 used one scale per 32 elements (g=32) plus GPTQ calibration. NVFP4 uses one FP8-E4M3 scale per 16 FP4-E2M1 elements. MXFP4/MXFP8 use one E8M0 scale per 32 elements ([[triton-docs]], tutorial 10).
- Native block-scaled tensor-core MMA (`tl.dot_scaled`) needs **CC 10.x (Blackwell)**. **H200 is Hopper (SM90)**, so NVFP4 there means dequant-to-bf16/fp8 in kernel, i.e. weight-only style. (inference from the tutorial's hardware requirement)

## Key ideas
- **MBU framing:** `MBU = params × bytes_per_param × tokens/s / peak_BW`. You can't change params or bandwidth, but you can change bytes/param (gpt-fast).
- **int8 weight-only (per-channel):** `y = (x @ W_int8.T.to(bf16)) * scale[n]`. The scale is applied once per output after the K reduction, which is nearly free.
- **int4 grouped:** the scale (and optionally zero point) changes every G elements of K, so it must be applied *inside* the K loop: per group, dequant → FMA, or accumulate a group partial and multiply it by the scale. torch.compile could not generate this fused kernel. gpt-fast used handwritten CUDA.
- **Accuracy:** int8 weight-only had no perceptible degradation. int4 did, mitigated by g=32 and GPTQ (gpt-fast).
- **Packing (inference):** two int4/FP4 values per byte. Repack offline so that one 16 B vector load yields a register layout that maps directly onto the MMA operand fragment or FMA lanes, avoiding shuffles. Interleaved layouts (as in Marlin-style kernels) exist for this reason. Do it in an untimed `prepare()`.
- **Dequant cost (inference):** at m=1 the ALU has a big surplus vs bandwidth, so dequant is hidden. As m grows (m=128–256), the dequant ALU work is amortized over more rows. But the kernel moves toward compute-bound, and on Hopper there's no FP4 MMA, so bf16 or FP8 wgmma after dequant is the ceiling.

## Numbers worth remembering
- gpt-fast, Llama-7B, A100-80GB @330 W, BS=1: fp16 compiled 107.0 → int8 WO 157.4 → int4 g32 GPTQ 202.1 → combined with speculative 244.7 tok/s. Llama-70B TP + int8: 55 tok/s.
- NVFP4: group 16, E4M3 scale. MX formats: group 32, E8M0 scale (Triton tutorial 10).
- NVFP4 scale layout in Triton's Blackwell tutorial: 5D `(M//32//4, K//VEC//4, 32, 4, 4)` so each MMA reads 128 contiguous scale rows. TMA-loaded as uint8 `[1, rep_m, rep_k, 2, 256]`.

## Code patterns
```python
# gpt-fast-style int8 weight-only linear (per-output-channel scale); compile fuses cast+scale into the GEMV
class WeightOnlyInt8Linear(torch.nn.Module):
    def forward(self, x):                                   # x: [m, K] bf16
        return F.linear(x, self.weight.to(x.dtype)) * self.scales   # weight: int8 [N, K]; scales: [N]
```
```python
# Triton: Blackwell block-scaled MMA (tutorial 10). NOT available on H200/SM90.
acc = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", acc)   # nvfp4/mxfp4
```
```python
# Hopper fallback (inference): in-register dequant inside the K loop
w_packed = tl.load(wq_ptr + offs)                    # uint8, 2 x fp4 per byte
lo, hi = w_packed & 0xF, w_packed >> 4               # unpack nibbles
w = lut_e2m1(lo, hi).to(tl.bfloat16) * scale_grp     # E2M1 -> bf16 via LUT/bit tricks, times the group scale
acc = tl.dot(x, tl.trans(w), acc)                    # or FMA for m=1
```

## Pitfalls
- A separate dequant pass destroys the benefit (gpt-fast: int8 without compile was worse than fp16).
- Group-wise scales inside the K loop add loads. Keep scales in a layout contiguous with the weight stream so they ride along in the same cache lines/TMA boxes. (inference)
- Don't apply weight-only quant to a problem graded at BF16 tolerance (`muse_bf16_proj`). It changes numerics. (inference)
- Rubin/Blackwell-specific scale preshuffles don't transfer to Hopper.

## Relevance to the serving track
- `muse_bf16_proj` is BF16. No quantization is allowed, but the *roofline logic* is the same: time = weight bytes / achieved BW.
- NVFP4 MoE on 8×H200 (inference): no FP4 tensor cores on SM90, so the kernel is "weight-only NVFP4". Stream 0.5 B/param + 1 E4M3 scale per 16 values (≈0.5625 B/param), dequant to bf16 (or FP8 E4M3 for FP8 wgmma) in registers, and accumulate in fp32. Small expert batches (decode/verify m per expert is tiny) mean it's bandwidth-bound like gpt-fast's GEMV. Fusion and a register-friendly offline repack in `prepare()` decide performance. See [[gemv-and-skinny-gemm]], [[split-k-and-stream-k]], [[wgmma]].
- The global scale (NVFP4 per-tensor FP32 scale) can be folded into the epilogue. (inference)

## Related
[[pytorch-gpt-fast-accelerating-genai-2]] · [[triton]] · [[triton-docs]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[wgmma]] · [[tma]] · [[speculative-decoding-shapes]] · [[benchmarking-kernels]]
