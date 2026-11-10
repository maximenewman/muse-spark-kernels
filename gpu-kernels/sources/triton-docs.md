---
title: Triton documentation (triton-lang.org)
type: source
tags: [gpu, triton, dsl, matmul, tma, autotune, hopper, blackwell, pdl, benchmarking]
source_url:
  - https://triton-lang.org/main/index.html
  - https://triton-lang.org/main/programming-guide/chapter-1/introduction.html
  - https://triton-lang.org/main/getting-started/tutorials/index.html
  - https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
  - https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  - https://triton-lang.org/main/python-api/triton.language.html
  - https://triton-lang.org/main/python-api/generated/triton.language.range.html
  - https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html
  - https://triton-lang.org/main/python-api/generated/triton.autotune.html
  - https://triton-lang.org/main/python-api/generated/triton.Config.html
  - https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
  - https://triton-lang.org/main/python-api/triton.language.extra.cuda.html
  - https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
  - https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py
---
# Triton documentation (triton-lang.org)

**TL;DR**
- Model: **"Blocked Program, Scalar Threads"**, in contrast to CUDA's "Scalar Program, Blocked Threads". You write one program instance per tile. The compiler handles coalescing, thread swizzling, prefetching, vectorization, tensor-core instruction selection, shared-memory allocation/sync, and async-copy scheduling.
- Hopper-relevant surface (docs, Nov 2026): TMA via `tl.make_tensor_descriptor` (device-side) or the host `TensorDescriptor.from_tensor`. Clusters via `triton.Config(num_ctas=…)` (SM90+). Per-loop pipelining via `tl.range(num_stages=…)`. PDL via `tl.extra.cuda.gdc_wait` / `gdc_launch_dependents`. `tl.dot_scaled` for block-scaled (MX/NVFP4-style) matmul.
- **`tl.range(warp_specialize=True)` is documented as Blackwell-only, and only for simple matmul loops.** On Hopper you get software pipelining, not compiler warp specialization (the persistent-matmul tutorial FIXME says the same).
- Low-level escape hatch: **Gluon** (a separate section with its own tutorials/API), plus `inline_asm_elementwise` and `ir_override` (load a hand-edited `.ttgir/.llir/.ptx`).
- `triton.testing.do_bench` flushes L2 before every run by zeroing a 256 MB buffer. `do_bench_cudagraph` does **not** flush.

## Key ideas
**Doc sections:** Getting Started (Installation, Tutorials) · Python API (`triton`, `triton.language`, `triton.testing`, Triton Semantics, `triton.language.extra.cuda`) · Gluon (Overview, Tutorials, Examples, API) · Triton MLIR Dialects · Programming Guide (Introduction, Related Work, Debugging Triton, Floating-Point Sanitizer "FpSan").

**Tutorials (in order):** 01 Vector Addition · 02 Fused Softmax · 03 Matrix Multiplication · 04 Low-Memory Dropout · 05 Layer Normalization · 06 Fused Attention · 07 Libdevice (`tl.extra.libdevice`) · 08 Group GEMM · 09 Persistent Matmul · 10 Block Scaled Matrix Multiplication. A PDL tutorial `11-programmatic-dependent-launch.py` exists in the tutorials tree (seen in a fork mirror; not in the rendered index I fetched).

**triton.language API groups (exact names):**
- Programming model: `tensor`, `tensor_descriptor`, `program_id`, `num_programs`, `map_elementwise`
- Linear algebra: `dot`, `dot_scaled`
- Memory: `load`, `store`, `make_tensor_descriptor`, `load_tensor_descriptor`, `store_tensor_descriptor`. **`make_block_ptr`/`advance` are no longer listed** on the current API page (the descriptor API replaces them).
- Indexing: `flip`, `where`, `swizzle2d`. Iterators: `range`, `static_range`
- Compiler hints: `assume`, `debug_barrier`, `max_constancy`, `max_contiguous`, `multiple_of`
- Atomics: `atomic_add/and/cas/load/max/min/or/poll/store/xchg/xor`
- Inline asm: `inline_asm_elementwise`. Debug: `static_print`, `static_assert`, `device_print`, `device_assert`, `expect_zero`
- Scan/sort: `associative_scan`, `cumsum`, `cumprod`, `histogram`, `sort`, `topk`, `gather`
- `triton.language.extra.cuda`: `gdc_wait` (block until the prior kernel's instructions completed) and `gdc_launch_dependents` (the next kernel may launch once all programs have called it or exited). The launch side is enabled with a `launch_pdl` launch kwarg (per the PDL tutorial; verify against your Triton version).

**`tl.range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False)`**
- `num_stages`: pipelines *most loads in the loop*. In contrast, the kernel-level `num_stages` pipelines only loads feeding `dot`.
- `loop_unroll_factor` (<2 means no unroll). `flatten` collapses the loop nest into a single loop for pipelining. `disallow_acc_multi_buffer`. `disable_licm` avoids long live ranges.
- `warp_specialize`: partitions memory/MMA/vector ops into async partitions (more warps). **Blackwell only, simple matmul loops.**

**`tl.make_tensor_descriptor(base, shape, strides, block_shape, padding_option='zero', fp6_format=None)`**
- `base` must be 16-B aligned. Leading strides must be multiples of 16 B. The last dim must be contiguous. 2–5 D only.
- On TMA-capable NVIDIA GPUs it builds a TMA descriptor, and loads/stores go through TMA hardware.
- Device-side descriptors need a global scratch allocation: `triton.set_allocator(alloc_fn)` with `alloc_fn(size, alignment, stream) -> int8 CUDA tensor`.

**`triton.Config(kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None)`**
- `num_ctas` = CTAs per cluster (SM90+ only). `maxnreg` → PTX `.maxnreg`. `ir_override` loads a user `.ttgir/.llir/.ptx/.amdgcn` file.

**`triton.autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False)`**
- Re-tunes when any `key` arg value changes. `prune_configs_by={'perf_model', 'top_k', 'early_config_prune'}`. `reset_to_zero` is needed for kernels that accumulate (atomics/split-K), because tuning runs the kernel once per config. `warmup`/`rep` are deprecated in favor of `do_bench`. `TRITON_PRINT_AUTOTUNING=1` prints the chosen config.

**Matmul tutorial (03).** One program per `[BLOCK_M, BLOCK_N]` C tile. **Grouped ordering** (`GROUP_SIZE_M=8`) improves L2 reuse. In the 9×9-block example, computing the first 9 outputs loads 90 blocks with row-major order vs 54 with grouped order. The page reports >10% on some hardware (220→245 TFLOPS on A100). It uses `tl.assume` for non-negative pids/strides, a masked K loop, an FP32 accumulator, and fuses the activation in the epilogue. The 16-entry CUDA config list spans BM∈{32..256}, BN∈{32..256}, BK∈{32,64,128}, stages 3–5, warps 2–8. **Its smallest BLOCK_M is 32, which is useless for m=1..16 without changes.**

**Persistent matmul tutorial (09).** Variants: naive, persistent (`grid = NUM_SMS`, `for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=...)`), TMA (host `TensorDescriptor.from_tensor(a, dummy_block)`, with the block shape set in an autotune pre_hook), TMA-persistent, a 2-CTA Blackwell variant, CLC (`clc=True`, CC ≥ 10.0), and device-descriptor persistent. Also `EPILOGUE_SUBTILE`: split the accumulator in two for the store, freeing SMEM for more stages. Code comment: "Hopper warpspec doesn't work with flatten." TMA/descriptors need CC ≥ 9.0. FP8 (`float8_e4m3fn`) needs CC ≥ 9.0. It fails on small-SMEM GPUs (RTX 4090). Profiling uses Proton (`proton.start(..., hook="triton")`, `proton.scope` with bytes/flops metadata).

## Numbers worth remembering
- Tutorial 03 FP16 at 4096³: cuBLAS ≈ 221.5 vs Triton ≈ 217.9 TFLOPS (hardware unspecified). Mid sizes swing both ways: 1792 → Triton 204.4 vs 172.9; 1920 → 162.6 vs 200.3.
- Grouped ordering: 90 → 54 block loads (9×9 example). Reported A100 220 → 245 TFLOPS.
- `do_bench` defaults: `warmup=25` ms, `rep=100` ms, `return_mode='mean'` (also `median/min/max/all`), optional `quantiles`. The L2 flush buffer is 256 MiB (`torch.int`, `cache.zero_()` before each timed call).
- `do_bench_cudagraph`: captures `n_repeat` unrolled calls in one graph, replays 10×, and divides by `n_repeat`. **No L2 flush.**

## Code patterns
```python
@triton.jit
def k(in_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    desc = tl.make_tensor_descriptor(in_ptr, shape=[M, N], strides=[N, 1],
                                     block_shape=[BM, BN])   # TMA descriptor (sm90+)
    v = desc.load([tl.program_id(0)*BM, tl.program_id(1)*BN])  # TMA load, OOB zero-padded
    desc.store([tl.program_id(0)*BM, tl.program_id(1)*BN], tl.abs(v))
triton.set_allocator(lambda size, align, stream: torch.empty(size, device="cuda", dtype=torch.int8))
```
```python
# Persistent tile loop (tutorial 09): grid = min(NUM_SMS, num_tiles)
for tile_id in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True):
    pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
    ...
```
```python
# PDL in Triton (pattern from the PDL tutorial; check your version)
if USE_GDC: tl.extra.cuda.gdc_wait()             # before reading the producer's outputs
...
if USE_GDC: tl.extra.cuda.gdc_launch_dependents() # lets the next kernel start its prologue (no memory-ordering guarantee)
kernel[grid](..., USE_GDC=True, launch_pdl=True)
```

## Pitfalls
- The autotuner benchmarks inside the first call. Never let it run inside a CUDA-graph capture or a timed region. Warm all shapes in `prepare()`. Autotune keyed on `M` re-tunes per m. (inference)
- `do_bench` mean is sensitive to outliers. Prefer `return_mode='median'` or quantiles. `do_bench_cudagraph` measures warm-L2 back-to-back replays, which is optimistic for cold-L2 harnesses. (inference)
- Descriptor constraints (16 B alignment, 16 B strides) apply to weights repacked in `prepare()`. Pad K/N rows as needed.
- Code tuned for A100 often underperforms on H100 (see [[modular-triton-and-python-edsls]]).
- Device-side descriptors allocate scratch through `set_allocator`. That allocation must be graph-safe (inference: allocate from a persistent buffer, not per call).

## Relevance to the serving track
- For m ≤ 32 the tutorial matmul configs (BM ≥ 32, tl.dot) waste most of the MMA tile. Write a dedicated small-m kernel. Options: BM=16 (the minimum `tl.dot` M on NVIDIA is 16, so pad m=1/8 rows via masked loads), or a pure-FMA GEMV with `tl.sum` for m=1. Add split-K for occupancy across 132 SMs. See [[triton]], [[gemv-and-skinny-gemm]], [[split-k-and-stream-k]]. (inference)
- The weight stream is what matters. TMA descriptors + `tl.range(num_stages=4..6)` give deep async prefetch of W tiles on H200 without hand-written mbarriers. (inference)
- PDL (`gdc_*`) can overlap your kernel's launch/prologue with the preceding kernel inside SGLang's graph. Whether the harness's single-op graph benefits is doubtful. See [[pdl]], [[cuda-graphs]]. (inference)
- `do_bench`'s 256 MB `zero_()` flush is the right cold-L2 model for H200's L2. Use it, or replicate it inside graph replay ([[benchmarking-kernels]]). (inference)

## Related
[[triton]] · [[tma]] · [[pdl]] · [[wgmma]] · [[cuda-graphs]] · [[benchmarking-kernels]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[swizzling]] · [[modular-triton-and-python-edsls]] · [[pytorch-gpt-fast-accelerating-genai-2]]
