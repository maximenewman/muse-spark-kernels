---
title: Roofline and Arithmetic Intensity
type: concept
tags: [gpu, roofline, arithmetic-intensity, gemm, gemv, decode, hopper]
source_url:
  - https://modal.com/gpu-glossary/perf/roofline-model
  - https://modal.com/gpu-glossary/perf/arithmetic-intensity
  - https://modal.com/gpu-glossary/perf/memory-bandwidth
  - https://modal.com/gpu-glossary/perf/memory-bound
  - https://www.nvidia.com/en-us/data-center/h200/
---
# Roofline and Arithmetic Intensity

**TL;DR**
- Arithmetic intensity (AI) = FLOPs / bytes moved. Roofline: perf ≤ min(peak FLOP/s, AI × BW). Ridge point = peak / BW = minimum AI to be compute-bound (Modal).
- Ridge (dense BF16): A100 156, **H100 SXM 295**, B200 281 FLOP/B (Modal). H200 (inference): ~989 dense TFLOPs / 4.8 TB/s ≈ **206 FLOP/B**.
- Small-m GEMM `out = x @ W.T` has AI ≈ m FLOP/B (weights dominate bytes) → m ≤ 32 is deeply memory-bound; m=128 is ~60% of H200 ridge; m=256 is at/over H200 ridge.
- Roofline ignores latency (Modal) – for 2–17 µs kernels, latency/ramp-up and launch overhead create a third "roof" below the bandwidth line; see [[memory-bandwidth-and-latency]].

## Key ideas
- Roofs per subsystem: Tensor Core roof ≫ CUDA-core roof; a skinny GEMM done on CUDA cores (FMA) can become compute-bound much earlier than the tensor-core ridge suggests (inference from Modal's "each subsystem has its own roofs").
- Kernel table (Modal): SAXPY AI 1/6; FP32 real FFT (5/32)·log N; SGEMM 2N³/16N² = N/8.
- Decode is memory-bound because weights reload per token; batching (larger m) raises FLOPs per weight byte "without adding latency" until compute-bound (Modal memory-bound page).
- Nsight Compute plots kernels on a roofline automatically.

## Numbers worth remembering
| System | TFLOP/s | TB/s | Ridge | Source |
|---|---|---|---|---|
| H100 SXM BF16 | 989 | 3.35 | 295 | Modal |
| H100 SXM FP8 | 1979 | 3.35 | 592 | Modal |
| B200 BF16 | 2250 | 8 | 281 | Modal |
| H200 SXM BF16 | 1979 *sparse* (≈989 dense, inference) | 4.8 | ≈206 (inference) | NVIDIA H200 page |

## Worked example: small-m projection (inference)
`out[m,n] = x[m,k] @ W[n,k]^T`, BF16 in/out, each tensor touched once:
- FLOPs = 2·m·n·k
- Bytes = (n·k + m·k + m·n)·2
- AI = m·n·k / (n·k + m·k + m·n) ≈ m when m ≪ n,k.

Illustrative n = k = 4096 (W = 32 MiB) on H200 (4.8 TB/s, 989 dense TFLOP/s):
| m | AI (FLOP/B) | t_mem @4.8 TB/s | t_compute @989 TF | bound |
|---|---|---|---|---|
| 1 | 1.0 | 6.99 µs | 0.03 µs | memory |
| 8 | 8.0 | 7.02 µs | 0.27 µs | memory |
| 16 | 15.9 | 7.05 µs | 0.54 µs | memory |
| 32 | 31.5 | 7.10 µs | 1.09 µs | memory |
| 128 | 120.5 | 7.43 µs | 4.34 µs | memory, compute must overlap |
| 256 | 227.6 | 7.86 µs | 8.69 µs | ≈ balanced / compute at peak |
n = k = 8192: AI 1.0 / 8.0 / 15.9 / 31.8 / 124.1 / 240.9; t_mem ≈ 28.0–29.7 µs.

Takeaways (inference):
- Bytes are mostly weights: for n=k=4096, x + out add (m·k + m·n)/(n·k) = 2m/4096 → <1% for m ≤ 16, 6.25% at m=128, 12.5% at m=256. The time floor is ≈ `n·k·2 / BW` for small m.
- At m=128–256 the tensor cores must run at a high fraction of peak *concurrently* with full-rate streaming; at peak dense wgmma rates m=256 is already compute-limited on H200. CUDA-core FMA paths (≈ tens of TFLOP/s) would be hopelessly compute-bound for m ≥ 32 – use [[wgmma]]/[[mma-ptx]] there.
- FP32 router as BF16 hi+lo split (inference): W_hi + W_lo doubles bytes to FP32 size (same as streaming FP32), and FLOPs double (x·W_hi + x·W_lo) or triple if x is also split (x_hi·W_hi + x_hi·W_lo + x_lo·W_hi). AI per byte stays ≈ m (or 1.5·m), so still memory-bound for small m.

## Code patterns
```python
def ai(m, n, k, bpe=2):           # bytes per element
    return 2*m*n*k / ((n*k + m*k + m*n) * bpe)
def floor_us(m, n, k, bw=4.8e12, bpe=2):
    return (n*k + m*k + m*n) * bpe / bw * 1e6
```

## Pitfalls
- Using sparse TFLOPS (NVIDIA headline numbers for H200) halves the apparent ridge correctly only if you divide by 2 for dense.
- Spec bandwidth is not achievable; real achieved HBM BW is lower (Veitner's transpose got 84% of 3.3 TB/s steady-state) – budget for it.
- Roofline says nothing about bytes-in-flight; a kernel can sit far below the slanted roof purely from insufficient concurrency.

## Relevance to the serving track
- (inference) For `muse_bf16_proj` the per-shape floor the judge quotes is exactly `bytes / 4.8 TB/s`. Beating SGLang means getting closer to that line: concurrency (bytes in flight), launch overhead, and for m ≥ 128 hiding MMA time under the stream.
- (inference) Separate kernel strategies by m: m ∈ {1,8,16} GEMV-ish (even CUDA cores or small mma tiles fine, focus purely on bandwidth); m ∈ {32,128,256} tensor-core skinny GEMM with TMA pipeline.

## Related
[[memory-bandwidth-and-latency]] · [[gpu-memory-hierarchy]] · [[modal-gpu-glossary]] · [[wgmma]] · [[mma-ptx]] · [[gtc25-s72683-memory]] · [[gemv-and-skinny-gemm]] · [[speculative-decoding-shapes]] · [[benchmarking-kernels]]
