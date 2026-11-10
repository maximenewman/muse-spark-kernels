---
title: "Accelerating Generative AI with PyTorch II: GPT, Fast (gpt-fast)"
type: source
tags: [gpu, llm-inference, decode, torch-compile, cuda-graphs, quantization, speculative-decoding, tensor-parallel, memory-bound]
source_url: https://pytorch.org/blog/accelerating-generative-ai-2/
---
# Accelerating Generative AI with PyTorch II: GPT, Fast

**TL;DR**
- Batch-1 LLaMA-7B decode in <1000 LOC of pure PyTorch: 25.5 → 107.0 (compile + static KV + CUDA graphs) → 157.4 (int8 weight-only) → 202.1 (int4 g32 + GPTQ) → 244.7 tok/s (all combined incl. speculative decoding). A100-80GB power-limited to 330 W.
- The naive baseline is **CPU-overhead bound**; `torch.compile(mode="reduce-overhead", fullgraph=True)` (which applies CUDA graphs) gives >4x.
- After that, decode is **weight-bandwidth bound**: every matmul at BS=1 is a GEMV. They measure it with MBU (Model Bandwidth Utilization). At 107 tok/s the MBU is 72%.
- The only remaining levers: fewer bytes per parameter (int8/int4 weight-only), fewer weight passes per token (speculative decoding, which verifies 8 draft tokens in one target pass), or more aggregate bandwidth (tensor parallel).
- Inductor-generated Triton GEMV kernels **beat cuBLAS** for these matrix-vector shapes. Fused int8 dequant needs compile. Without fusion, int8 is *slower* than fp16.

## Key ideas
- **Overhead-bound regime.** The CPU (single-threaded Python) cannot issue kernels fast enough, so the GPU idles between tiny kernels. The fix is to submit a whole decode step at once, i.e. CUDA graphs via `mode="reduce-overhead"`. `fullgraph=True` asserts there are no graph breaks. See [[cuda-graphs]].
- **Static KV cache.** Allocate the max-length KV cache once and mask unused positions in attention. A growing cache means realloc+copy and dynamic shapes, which break CUDA graphs. This is the canonical "static buffers" rule for graph capture.
- **Separate compile strategies for prefill vs decode.** Prefill has a variable prompt length (dynamic). Decode is fully static once the KV cache is static.
- **Why compile wins on kernels too.** For BS=1 decode, compile generated *every* kernel from scratch (matmuls and attention) and beat cuBLAS and FlashAttention2. The reason: matrix-vector work is purely memory-bandwidth bound, "well within the range of compilers to automatically generate."
- **MBU** = (params × bytes/param × tokens/s) / peak bandwidth. 7B × 2 B × 107 tok/s / 2 TB/s ≈ 72%. The post notes "even just copying memory struggles to break 85%". So at 72% you might only get another ~10% without changing the problem.
- **int8 weight-only.** Only the weights are quantized. Compute is still bf16. One scale per output row. The dequant must be **fused** into the matmul to pay off. See [[weight-only-quantization]].
- **int4.** Accuracy drops, so they use (a) group-wise scales, one per 32 elements (groupsize=32), and (b) GPTQ calibration (implemented via `torch.export`). torch.compile could **not** generate the fused int4-dequant GEMV, so they used handwritten CUDA kernels in PyTorch.
- **Speculative decoding.** The draft model generates 8 tokens, then the verifier processes all 8 in parallel and discards the mismatches. Output is mathematically identical to the target model. The speedup depends on draft/target agreement and the text. About 50 LOC. See [[speculative-decoding-shapes]].
- **Tensor parallelism** increases the MBU denominator (aggregate HBM bandwidth across GPUs). It is ~150 LOC (`tp.py`), composes with compile, and needs no model changes.
- **AMD.** Since every decode kernel is Inductor→Triton, the same stack runs on MI250x via Triton's AMD backend.

## Numbers worth remembering
| Step | Model / HW | tok/s |
|---|---|---|
| Baseline eager | Llama-7B, A100-80GB @330 W, BS=1 | 25.5 |
| + torch.compile reduce-overhead + static KV cache | same | 107.0 (MBU 72%) |
| + int8 weight-only (fused via compile) | same | 157.4 (~+50%) |
| int4 (g=32) + GPTQ + handwritten CUDA kernels | same | 202.1 |
| Combined (compile + int4 + speculative) | Llama-7B | 244.7 (chart); conclusion text says 241 |
| int8 on 1 GCD of MI250x | Llama-7B | 102.5 |
| Speculative: CodeLlama-34B target + CodeLlama-7B draft (code gen) | — | ~2x |
| Speculative: Llama-7B + TinyLlama-1B | — | ~1.3x |
| TP + int8 | Llama-70B | 55 |
| TP + compile + int4 + speculative | Llama-70B | 80 |
- Copy-kernel ceiling ≈ 85% of peak HBM bandwidth (A100, as stated).
- LOC: model.py 244 + generate.py 371 + tp.py 151 = 766.

## Code patterns
```python
# Compile only the per-token decode step; reduce-overhead => CUDA graphs; fullgraph => error on graph breaks
decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
```
- Static KV cache: buffers `[B, H, max_seq_len, D]` registered once, written with `k_cache[:, :, input_pos] = k_val` (an index_put at the dynamic position *tensor*, not a Python int). Attention uses a precomputed causal mask sliced by `input_pos`. Shapes never change, so the graph is replayable. (The pattern is from the gpt-fast repo linked by the post. The post shows it as an image.)
- int8 weight-only linear (gpt-fast `quantize.py`, shown as an image in the post). It is essentially `F.linear(x, self.weight.to(x.dtype)) * self.scales`, where `weight` is int8 `[N,K]` and `scales` is per-output-channel. Inductor fuses the cast and the scale into a single GEMV kernel.

## Pitfalls
- int8 **without** compile/fusion is slower than fp16. A separate dequant kernel materializes bf16 weights, which *adds* memory traffic.
- int4 needed hand-written fused kernels. Compiler-generated code was not enough for sub-byte unpacking.
- Speculative decoding's speedup is data-dependent, so benchmark it on representative text.
- Dynamic KV length kills CUDA graphs. Everything in the captured step must have static shapes and addresses.

## Relevance to the serving track
- `muse_bf16_proj` with m=1 is exactly the gpt-fast regime: a GEMV-like op, weight-bandwidth bound. Use MBU-style accounting, `N·K·2 B / t_kernel` vs H200 HBM peak, to know how close you are. The post's empirical ceiling of ~85% of peak for copies is a good sanity cap. (inference)
- m ∈ {8,16,32} equals 8-token verify blocks × batch (see [[speculative-decoding-shapes]]). The weight-bytes cost is the same as m=1, so time should be ~flat from m=1 to m=32 if the kernel is truly bandwidth bound. If it is not flat, you're compute- or latency-limited in the inner loop. (inference)
- Harness timing is CUDA-graph replay with the ~4 µs graph cost subtracted. This removes exactly the CPU overhead gpt-fast fought, so only device time matters. A kernel that "wins" eagerly due to fewer launches does not necessarily win here. (inference)
- Weight-only quant is not allowed for a BF16 problem (correctness tolerance), but the NVFP4 MoE problems are the same idea taken further. See [[weight-only-quantization]]. Repacking in `prepare()` is the analogue of gpt-fast's offline `quantize.py`. (inference)
- Inductor-Triton beating cuBLAS on GEMV is a reminder that cuBLAS is not the floor for small m. A hand-written split-K / GEMV kernel can win. See [[gemv-and-skinny-gemm]], [[split-k-and-stream-k]]. (inference)

## Related
[[cuda-graphs]] · [[weight-only-quantization]] · [[speculative-decoding-shapes]] · [[benchmarking-kernels]] · [[triton]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[triton-docs]]
