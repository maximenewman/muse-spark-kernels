---
title: Triton (for small-m serving GEMMs)
type: concept
tags: [gpu, triton, small-m, gemv, split-k, autotune, hopper, tma, decision]
source_url:
  - https://triton-lang.org/main/index.html
  - https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls
  - https://pytorch.org/blog/accelerating-generative-ai-2/
---
# Triton (for small-m serving GEMMs)

**TL;DR**
- Triton = one program per tile ("Blocked Program, Scalar Threads"). The compiler owns coalescing, SMEM allocation/sync, async-copy pipelining, and tensor-core instruction selection ([[triton-docs]]). You own tile shapes, grid, split strategy, masking, and data layout.
- Memory-bound GEMV/skinny GEMM is Triton's sweet spot. Inductor's auto-generated Triton GEMV beat cuBLAS for BS=1 decode ([[pytorch-gpt-fast-accelerating-genai-2]]). The ~20% Triton-vs-CUDA gap claimed by [[modular-triton-and-python-edsls]] concerns compute-bound Hopper kernels (wgmma/TMA/warp-specialization).
- Hopper features exposed: TMA via tensor descriptors, clusters via `num_ctas`, per-loop `num_stages`, PDL via `gdc_wait`/`gdc_launch_dependents`, and wgmma is chosen by the compiler for `tl.dot`. **Not exposed on Hopper:** compiler warp specialization (documented Blackwell-only), explicit mbarrier/TMA-multicast control, setmaxnreg producer/consumer register splitting. For those, use Gluon or CUDA/CuTe.
- For the serving track: write a per-m kernel family with **split-K** for m ≤ 32 to fill 132 SMs. Pre-tune configs per m and hard-code them (no autotune at runtime). Pre-allocate partial/semaphore buffers.

## Key ideas
### Choosing Triton vs CUDA vs CuTe for small-m GEMM (inference, grounded in sources above)
| Situation | Pick | Why |
|---|---|---|
| m=1..32 BF16, memory-bound, need fast iteration | **Triton** | Bandwidth roofline reachable with masked loads + split-K. Easy per-m specialization. Inductor-style GEMV already beats cuBLAS (gpt-fast). |
| Need W loads issued *before* the PDL wait, custom in-kernel split-K reduction with a self-resetting counter, exact control of load width / `ld.global.nc` / L2 eviction hints | CUDA C++ (or Triton + `inline_asm_elementwise` / atomics) | Triton atomics + `gdc_*` cover much of this, but you can't pin instruction scheduling. |
| m=128/256, getting compute-relevant, want wgmma + TMA + warp-specialized pipeline on Hopper | **CuTe/CUTLASS** (or CuTe DSL) | Triton's Hopper path is software-pipelined, not warp-specialized ([[triton-docs]]). See [[wgmma]], [[tma]]. |
| NVFP4 MoE (Blackwell block-scaled) | Triton `tl.dot_scaled` (tutorial 10) as a baseline. CUTLASS for peak. | Block-scaled MMA exposed in both. Fine layout control favors CUTLASS. |

### Small-m tiling in Triton (inference)
- `tl.dot` operands need ≥16 in each dim on NVIDIA. For m < 16, either pad rows with masked loads (`other=0`) into a BM=16 tile, or skip `tl.dot` and do FMA: `acc += tl.sum(x[None,:] * w, axis=1)` (GEMV form, W tile `[BN, BK]`). At m=1 the FMA form avoids wasting 15/16 of the MMA. At m=8–32 `tl.dot` with BM=16/32 is usually better.
- Weight layout: W is `[N,K]` row-major (`out = x @ W.T`), so K is contiguous. That's ideal: each program streams a `[BN, BK]` slab with 16 B-vectorized loads along K. Repack in `prepare()` if a different blocking helps (e.g. interleave for 128-bit loads), as long as the descriptor alignment rules hold ([[tma]]).
- Grid sizing: number of CTAs ≈ (N/BN) × SPLIT_K. Aim for ≥ 1–2 waves over 132 SMs with enough bytes in flight per SM. For small N·K, split-K is the only way to get parallelism. See [[split-k-and-stream-k]], [[gemv-and-skinny-gemm]].
- Pipelining: `num_stages` 3–6 on the K loop, or `tl.range(..., num_stages=…)` to pipeline non-dot loads too ([[triton-docs]]).

### Split-K in Triton (inference; the pattern is standard, the APIs are documented)
- **Atomic variant:** each (pid_n, pid_k) computes a partial `[BM, BN]` over its K slice, then `tl.atomic_add(out_fp32_ptr + offs, acc, mask=...)`. Needs a zeroed fp32 buffer (memset node, or zero inside the previous kernel) and a final cast to bf16 (an extra kernel, or a last-arriver cast). Nondeterministic summation order.
- **Two-pass variant:** write partials to `ws[SPLIT_K, M, N]` fp32, then a reduction kernel. Deterministic, but it adds a graph node.
- **Last-CTA variant:** after writing partials, `tl.atomic_add(counter_ptr + tile, 1)` with acq_rel semantics. The CTA that sees `SPLIT_K-1` reduces the partials, writes bf16, and resets the counter to 0 (so the graph replays cleanly). Single kernel, deterministic order.
- Autotuning a kernel that accumulates requires `reset_to_zero=[...]` in `triton.autotune` ([[triton-docs]]).

### Autotune configs
- `triton.Config(kwargs, num_warps, num_stages, num_ctas, maxnreg, pre_hook, ir_override)`, with `triton.autotune(configs, key=[...], prune_configs_by=..., reset_to_zero=..., do_bench=..., cache_results=...)`.
- Tutorial 03 configs start at BM=32 (useless for decode). Build your own small-m space: BM∈{16,32} (or FMA at m=1), BN∈{16,32,64,128}, BK∈{64,128,256,512}, SPLIT_K∈{1,2,4,8,16}, num_warps∈{2,4,8}, num_stages∈{2..6}. (inference)
- Tune offline under the real harness's conditions (graph replay, cold L2). Then **freeze the chosen config per (m, N, K)** in a dict and call the bare `@triton.jit` kernel. Autotune's default `do_bench` *does* flush L2 (256 MB `zero_()`), which matches the cold-L2 rules, but it times eager launches. (inference)

## Code patterns
```python
@triton.jit
def skinny_splitk(x_ptr, w_ptr, out_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1)        # grid = (cdiv(N,BN), SPLIT_K)
    rm = tl.arange(0, BM); rn = pid_n*BN + tl.arange(0, BN)
    k_per = tl.cdiv(K, SPLIT_K); k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), tl.float32)
    for kk in tl.range(k0, k0 + k_per, BK, num_stages=4):     # pipelined K loop over this split
        rk = kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None]*K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.)
        w = tl.load(w_ptr + rn[:, None]*K + rk[None, :], mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.)
        acc = tl.dot(x, tl.trans(w), acc)                      # BM>=16: rows >= M are zero padding
    tl.atomic_add(out_ptr + rm[:, None]*N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))  # fp32 out, pre-zeroed
```
(Illustrative sketch, not benchmarked. Masks on `k0+k_per` boundaries are simplified.)

## Pitfalls
- JIT compile + autotune on first call: must happen before graph capture.
- `tl.atomic_add` into bf16 is not what you want (precision/perf). Accumulate in fp32 then cast.
- Masks on the weight load in the hot loop cost instructions. Pad N/K in `prepare()` so the steady-state loop is unmasked (EVEN_K constexpr). (inference)
- Code tuned on A100 can underperform on H100/H200 ([[modular-triton-and-python-edsls]]). Re-tune on the target.
- Router is FP32 (per problem statement). Don't silently cast to bf16 `tl.dot`. Use `input_precision`/fp32 FMA as appropriate. (inference)

- **Runtime vs `tl.constexpr` split-K arguments change the best config.** On H100, `('gemv', 4 rows, BK 1024)` for qkv_gate m=1 ran 12.9 µs with runtime `K_SPLIT`/`SPLIT_K` and 15.7 µs with them as constexpr. Tune and ship the same form ([[findings-log]]).
- **Triton 3.6 on H100 didn't beat cuBLAS at m = 128/256** for BF16 projections in either orientation (swap-AB or `x @ Wᵀ`). The standard orientation only won for the FP32 router via a hi/lo split, where cuBLAS needs three kernels ([[findings-log]]).
- Compile time dominates a sweep. Precompile configs in a process pool so they land in the on-disk cache, then time serially ([[rorqual-h100-workflow]]).

## Relevance to the serving track
- Floors of 2–17 µs mean the kernel is a few µs of HBM streaming. Launch prologue, grid-tail imbalance, and reduction epilogue are a large fraction. Triton is fine for the body. Watch the tail (wave quantization over 132 SMs) and the split-K reduction cost. (inference)
- Per-shape graph capture lets you pick a distinct Triton specialization per m at Python level for free ([[cuda-graphs]]).
- Validate with [[benchmarking-kernels]] methodology (graph replay + L2 flush + median), not `do_bench` means.

## Related
[[triton-docs]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[tma]] · [[wgmma]] · [[pdl]] · [[cuda-graphs]] · [[benchmarking-kernels]] · [[weight-only-quantization]] · [[modular-triton-and-python-edsls]] · [[pytorch-gpt-fast-accelerating-genai-2]]
