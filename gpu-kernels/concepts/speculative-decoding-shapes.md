---
title: Speculative decoding and the GEMM shapes it produces
type: concept
tags: [gpu, speculative-decoding, decode, gemm-shapes, small-m, memory-bound, serving]
source_url:
  - https://pytorch.org/blog/accelerating-generative-ai-2/
---
# Speculative decoding and GEMM shapes

**TL;DR**
- A draft model proposes k tokens. The target ("verifier") model scores all k **in one forward pass**, accepts the longest matching prefix, and resamples at the first mismatch. Output is identical to plain target decoding. gpt-fast uses k=8 ([[pytorch-gpt-fast-accelerating-genai-2]]).
- In that verify pass, every linear layer sees **m = k × batch** rows instead of `batch`: 8 × {1,2,4,16,32} = {8,16,32,128,256}. These are exactly `muse_bf16_proj`'s non-decode m values. m=1 is plain decode. (inference matching the problem statement)
- Weight bytes are identical for m=1 and m=8..32, so a bandwidth-bound kernel should take ~the same time. Verifying 8 tokens costs about one decode step. That is the entire reason speculative decoding pays (gpt-fast: "in order to generate 100 tokens, we must load our weights 100 times").
- Arithmetic intensity grows linearly with m: ≈ m FLOP/B for BF16 GEMM with weight-dominated traffic. Even m=256 is still under the H200 ridge for BF16, so it's bandwidth-dominated but with heavier MMA work. (inference)

## Key ideas
- **Speedup is data-dependent:** CodeLlama-34B + 7B draft gives ~2x on code. Llama-7B + TinyLlama-1B gives only ~1.3x (gpt-fast). It's a win when draft time + verify time < time to generate the accepted tokens serially.
- **Why the verify pass is cheap (inference):** time(m) ≈ max(W_bytes / BW, 2·m·N·K / FLOPS) + overheads. For BF16 with m ≤ ~100s the first term dominates, so time(8) ≈ time(1). The kernel must actually achieve this. A GEMM tuned for m≥128 tiles (BM=128) wastes MMA at m=8 and may lack parallelism, and a GEMV kernel may not scale to m=32 (register pressure: m accumulators per output).
- **Attention in verify** becomes a small "prefill-like" chunk of k queries against the KV cache (causal within the k block). That's relevant to the attention problems. (inference)
- **Tree/EAGLE-style speculation** (not in source) changes k per step. In a CUDA-graph server, the token count is padded to captured sizes, which is why the server captures a fixed set of m values. (inference)

## Numbers worth remembering
- Draft length 8 in gpt-fast.
- ~2x (CodeLlama-34B/7B, code) and ~1.3x (Llama-7B/TinyLlama-1B) end-to-end tok/s gains.
- gpt-fast combined compile + int4 + speculative: 244.7 tok/s on Llama-7B (A100). Llama-70B with TP added: 80 tok/s.

## Code patterns
```python
# Shape bookkeeping for one verify step (inference)
k = 8                                   # draft tokens per sequence
m = k * batch                           # rows hitting every target-model linear
out = x.view(m, K) @ W.T                # W: [N, K] identical cost to decode in weight bytes
```

## Pitfalls
- Benchmarking only m=1 hides a kernel that degrades at m=16/32. The harness scores all m. (inference)
- m=128/256 may want a different kernel (tensor-core tiles, less split-K) than m≤32 (GEMV/split-K). Dispatch per m at capture time ([[cuda-graphs]]). (inference)
- Acceptance depends on text, so server-level gains vary. The kernel-level target is just "flat time from m=1 to m=32".

## Relevance to the serving track
- Design target (inference): `t(m) ≈ t(1)` for m ∈ {8,16,32}, and the minimum extra time for {128,256}. Concretely, one kernel family with BM ∈ {16,32} + split-K for m ≤ 32 (see [[triton]], [[split-k-and-stream-k]], [[gemv-and-skinny-gemm]]) and a tensor-core tile (BM=64/128, wgmma) for m ≥ 128 ([[wgmma]], [[tma]]).
- Rows from different sequences in a verify block are independent. No special masking in the GEMM. m is known at capture.

## Related
[[pytorch-gpt-fast-accelerating-genai-2]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[triton]] · [[cuda-graphs]] · [[weight-only-quantization]] · [[benchmarking-kernels]] · [[wgmma]]
