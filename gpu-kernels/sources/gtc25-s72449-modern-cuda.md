---
title: "GTC25 S72449: 1,001 Ways to Write CUDA Kernels in Python"
type: source
tags: [gpu, cuda, python, cuda-python, numba, cupy, gtc]
source_url:
  - https://www.nvidia.com/en-us/on-demand/session/gtc25-s72449/
  - https://www.besthub.dev/articles/how-to-write-cuda-kernels-in-python-insights-from-nvidia-gtc-2025-c4bd09f64da6
---
# GTC25 S72449: "1,001 Ways to Write CUDA Kernels in Python"

> **Title correction:** the task brief called this "Modern GPU Programming with CUDA". The On-Demand page and search results identify S72449 as **"1,001 Ways to Write (Accelerate Python with) CUDA Kernels in Python"**, presented by **Leo Fang** (NVIDIA Python CUDA tech lead). The On-Demand page renders no abstract in a non-JS fetch. The content below comes from search snippets and one secondary write-up (besthub.dev / Infra Learning Club). **I did not see the talk or slides.**

**TL;DR**
- Topic: authoring high-performance CUDA kernels from inside Python, covering memory management, thread organization, and optimization techniques (NVIDIA session blurb via search).
- Tools covered (per the secondary write-up): **Numba** `@cuda.jit`, **CuPy** custom elementwise/reduction kernels, **PyTorch C++/CUDA extensions**, and **cuda-python** with its subpackages: `cuda.core` (Pythonic runtime), `cuda.bindings` (low-level CUDA C API bindings), `cuda.cooperative` (block/warp-level sort/scan/reduce/transform, CUB-style), and `cuda.parallel` (device-wide CUB-style algorithms). Also **nvmath-python** (math libs). Triton is not mentioned in the write-up.
- Case study: a **segmented reduction**. C++ uses one block per segment, templated items-per-thread, per-thread partials, and a shared-memory tree reduce. The Python `@cuda.jit` version mirrors it with shared arrays.
- The Python path ultimately JIT-compiles to PTX/cubin at runtime and launches via `cuLaunchKernel`. Python is a host-side convenience, not a different execution model.
- Install: `pip install -U cuda-python` (or `cuda-python[all]`), or `conda install -c conda-forge cuda-python`.

## Key ideas
- NVIDIA's 2025 stance is "no one tool". It offers many Python entry points at different abstraction levels: library calls (nvmath), CUB-style building blocks (`cuda.cooperative`), Numba SIMT kernels, raw source via CuPy/`cuda.core` + NVRTC, and (elsewhere at GTC25) CUTLASS Python / CuTe DSL and cuTile ([[modular-triton-and-python-edsls]]).
- The underlying programming knowledge is unchanged: thread/block/grid hierarchy, memory hierarchy, execution model.

## Numbers worth remembering
- None technical. The write-up gives no kernel timings.

## Pitfalls
- The secondary source is thin. Verify any detail against the recording before relying on it.
- JIT-in-Python paths compile on first call. In a CUDA-graph-timed harness, all compilation must happen in `prepare()`/warmup, never in the captured call. (inference)

## Relevance to the serving track
- Practical use: `cuda.core`/NVRTC or CuPy `RawKernel` let you ship a hand-written CUDA kernel from a Python `prepare()` without a build step, which is handy if the harness is Python-only. Compile and load modules before capture. (inference)
- `cuda.cooperative` block reductions are relevant to the reduction tail of split-K GEMV ([[gemv-and-skinny-gemm]], [[split-k-and-stream-k]]). (inference)

## Related
[[sanglard-gpu-history]] · [[modular-triton-and-python-edsls]] · [[triton]] · [[cuda-graphs]]
