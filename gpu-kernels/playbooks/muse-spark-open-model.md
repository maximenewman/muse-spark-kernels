---
title: Muse Spark Open — model facts as served on 8×H200
type: playbook
tags: [muse, challenge, model, serving]
source_url:
  - raw/okc/problem-*.txt   # `okc show <problem>`, fetched 2026-11-08
  - raw/okc/serving.json    # GET /api/v1/serving
  - raw/okc/llms.txt        # https://openkernelchallenge.dev/llms.txt
---
# Muse Spark Open — model facts as served on 8×H200

**TL;DR**
- Meta's MoE model: **816B total, 42B active** (`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`), **256 routed experts, top-8**, NVFP4 expert weights, BF16 dense projections, FP32 router, FP8 E4M3 KV cache.
- **62 layers**: 46 sliding-window (2,048-token window, RoPE θ=500000) and 16 full-attention (no RoPE). Inferred from the per-shape weights 46 and 16 in the attention and layer problems, which sum to 62, matching the projection problem's weight of 62.
- Served with **TP8 attention / EP8 experts**, DFlash speculative decoding with **8 draft tokens**, so a verify step runs m = 8 × batch tokens.
- Where a decode step's per-GPU time goes in stock SGLang (the kernel track's baselines): **MoE 13.45 ms**, dense projections 4.83 ms, attention 3.77 ms; the whole layer comes to 22.62 ms. **The experts take about 60% of decode.**

## Dimensions (per GPU under TP8/EP8)
| Thing | Value |
|---|---|
| hidden | 8192 |
| Query heads / KV heads per GPU | 16 / 2 (GQA 8:1), head_dim 64 |
| qkv_gate weight | BF16 [2304, 8192] = q 16×64 + k 2×64 + v 2×64 + output gate 16×64, fused |
| o | BF16 [8192, 1024] (row-parallel, then all-reduce) |
| pooler | BF16 [4096, 8192]: hidden → MoE width, replicated on every rank |
| unpooler | BF16 [8192, 4096]: MoE width → hidden. At ≤8 tokens only this GPU's slice (rows 0–1023) |
| router | FP32 [256, 8192] + FP32 bias (used only to choose experts). Scores = `sigmoid(0.177 · h @ router.T) + bias`, top-8 |
| Experts | 256 total, 32 local per GPU (EP8), d_hidden 4096, d_expert 4096, SwiGLU |
| Expert weights | NVFP4: packed E2M1 `uint8 [E, 2I, H/2]` (w13 = gate then up rows), FP8 E4M3 block scales every 16 elements, FP32 global scale per expert |
| Post-expert norm | `rmsnorm(y * (1 + post_norm_w))` per expert output, *before* the routing weight and sum |
| KV cache | FP8 E4M3, page_size 1 in serving, scales = 1 (unset by SGLang) |
| Attention | QK-RMSNorm, q scaled by 5.473, sm_scale 1/8, output = `rmsnorm(a) * sigmoid(gate)` |

## One decoder layer (from `okc show muse_decode_layer`)
```python
# x: attention input [T, 8192] BF16   s: residual stream FP32
q, k, v, g = split(x @ qkv_gate.T);  QK-norm, RoPE, write fp8(k), fp8(v)
a = (rmsnorm(attention(q)) * sigmoid(g)) @ o.T     # all-reduce
s = α1 * s + β1 * rmsnorm(a);  h = rmsnorm(s) * (1 + w)
experts = top8(sigmoid(0.177 * h @ router.T) + bias)
y = local experts(rmsnorm(h @ pooler.T) * (1 + w))  # all-reduce
f = y @ unpooler.T   # T ≤ 8: [y @ unpooler[:1024].T, 0, …, 0]
s = α2 * s + β2 * rmsnorm(f * (1 + w))
return rmsnorm(s) * (1 + w), s, experts
```
Notable: the residual stream is **FP32**, and the residual updates are gated (α, β). The MoE runs at a narrower width (4096) reached through the pooler/unpooler projections.

## Shapes a decode/verify step produces
- Decode: T = batch ∈ {1, 16, 32}. Verify (DFlash, 8 draft tokens): T = 8 × batch ∈ {8, 128, 256}.
- Contexts are around 80K tokens (attention problem). Prefill: 753 new tokens on an 80K conversation, and 16K-token chunks on the first turn (`--chunked-prefill-size 16384`).
- So every dense GEMM is skinny (m ≤ 256) and weight-streaming → [[gemv-and-skinny-gemm]], [[roofline-and-arithmetic-intensity]].

## SGLang kernels stock uses (from the launch args)
- MoE: `--moe-runner-backend marlin` with `--quantization modelopt_fp4`, i.e. Marlin W4A16-style kernels on NVFP4 weights. H200 (Hopper) has no native FP4 tensor cores, so the weights are dequantized in-kernel.
- Attention: FA3, with the query rounded to FP8 (per the rules: "stock's FP8 query in FA3 attention stays as it is").
- Decode runs under CUDA graphs up to bs 32 (`--cuda-graph-max-bs-decode 32`) → [[cuda-graphs]].

## Related
[[serving-track-playbook]] · [[speculative-decoding-shapes]] · [[weight-only-quantization]] · [[gemv-and-skinny-gemm]] · [[tma]] · [[wgmma]]
