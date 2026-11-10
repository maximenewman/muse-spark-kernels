---
title: Benchmarking GPU kernels (µs-scale, serving-style)
type: concept
tags: [gpu, benchmarking, cuda-graphs, l2-flush, clock-locking, methodology, serving]
source_url:
  - raw/hidden-treasures-llms/3-GEMM/profile.sh
  - raw/hidden-treasures-llms/3-GEMM/src/structs_n_stuff.cuh
  - raw/hidden-treasures-llms/3-GEMM/src/cublas_benchmark.cu
  - https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
  - https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py
  - https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
  - https://docs.pytorch.org/docs/2.15/notes/cuda.html
  - https://pytorch.org/blog/accelerating-generative-ai-2/
---
# Benchmarking GPU kernels (µs-scale, serving-style)

**TL;DR**
- Mirror the judge: **CUDA-graph replay, cold L2, per-shape, subtract the fixed graph cost, report median.** Eager timing of a 2–17 µs kernel measures CPU launch overhead and Python, not the kernel (gpt-fast's 25.5 → 107 tok/s was pure overhead removal: [[pytorch-gpt-fast-accelerating-genai-2]]).
- **L2 flush:** Triton's `do_bench` zeroes a **256 MiB** buffer (`cache.zero_()`) before every timed call. That exceeds H200 L2, so weights are re-read from HBM. `do_bench_cudagraph` does **not** flush, so it's warm-L2 and optimistic.
- **Clock locking:** the matmul-playground `profile.sh` runs `nvidia-smi --lock-gpu-clocks=1590` and `--lock-memory-clocks=5000`, then resets both at the end. Locking removes boost/thermal variance. Use the *same* locked clocks for A/B comparisons, and note that memory clock matters most for memory-bound kernels.
- **Paired, interleaved A/B with medians.** Alternate candidate and baseline in the same process/graph, many reps, compare medians or ratios, not single means.

## Key ideas
### What the seed repo does (3-GEMM, read from source)
- `profile.sh`: lock GPU clock to 1590 MHz and memory clock to 5000 MHz → run `./build/runner <kernel_id> 50 M N K` for kernels 1–6 and cuBLAS (id 99) at 1024…16384 squared → `nvidia-smi --reset-gpu-clocks` / `--reset-memory-clocks`. Needs root (README: "run as sudo so you can set the GPU and DRAM clock speeds").
- `KernelLogger`: `cudaEventRecord(start, 0)` → launch → `cudaEventRecord(stop, 0)` → `cudaEventSynchronize` → `cudaEventElapsedTime`, then **`sleep_for(100 ms)` between runs** (lets clocks/thermals settle). It reports the **mean** of runs, as GFLOPS = 2MNK / t. One untimed warmup call. **No L2 flush.** Fine for large compute-bound squares (8192³). Wrong for µs kernels: per-launch events on the legacy stream include launch latency, and a warm L2 inflates memory-bound results.

### Triton `do_bench` (source)
- Run once, sync. Estimate time from 5 runs (each preceded by `clear_cache`). `n_warmup = max(1, warmup_ms/est)`, `n_repeat = max(1, rep_ms/est)`. Defaults: warmup 25 ms, rep 100 ms.
- Per rep: zero grads → `clear_cache` (256 MiB `zero_()`) → event start → `fn()` → event end. Summarize with `return_mode` ∈ {mean (default), median, min, max, all} or `quantiles=[0.2,0.5,0.8]`.
- Caveat (inference): each rep is an eager launch, so CPU launch overhead can leak into the event interval when the kernel is shorter than launch latency. The flush kernel right before each launch partially hides it, since the GPU is busy while the CPU enqueues.
- `do_bench_cudagraph`: captures `n_repeat` back-to-back calls into one graph, replays 10×, and divides by `n_repeat`. No flush. Measures warm-L2 steady-state throughput including inter-kernel overlap.

### Recommended harness for this competition (inference)
1. In `prepare()`: repack weights, allocate all outputs/workspaces, JIT/compile/autotune every shape, and run eager warmups.
2. For each m: capture `graph_k = [flush_kernel?, my_op(x_m)]`, or capture just `my_op` and do the flush outside the timed interval but before replay, matching the judge's definition.
3. Time with events around `graph.replay()` on a non-default stream. Many reps. Take the **median**. Subtract an **empty/no-op graph's replay time** measured the same way (the judge's ~4 µs fixed cost) to compare against the floors.
4. Cold L2: write ≥2× L2-size of scratch (e.g. 256 MiB `zero_()` like Triton) between replays. Also rotate among several copies of x if inputs could stay cached. Weights are the dominant traffic, and with a single weight copy the flush is what makes them cold.
5. **Paired comparison:** interleave baseline (`x @ W.T` via cuBLAS) and candidate replays (ABAB…), and report the median of per-pair ratios. This cancels clock drift. Lock clocks if you have root; otherwise warm the GPU to a steady state first.
6. Sanity: compute achieved bandwidth = (N·K·2 + m·K·2 + m·N·2) B / t and compare to H200 peak. gpt-fast notes even plain copies struggle to exceed ~85% of peak (A100), so a claimed >90% suggests a broken flush or measurement.
7. Cross-check with Nsight Systems (graph node durations) or `ncu --cache-control all --clock-control base`. ncu flushes caches and locks to base clocks by default, so its absolute times differ from boost-clock replay.

## Numbers worth remembering
- Triton L2 flush buffer: 256 MiB (`torch.int`, 67,108,864 elements), zeroed before each rep.
- `do_bench` defaults: warmup=25 ms, rep=100 ms, return_mode='mean'.
- matmul-playground locks: GPU 1590 MHz, memory 5000 MHz (tuned for its T4-era dev setup, not H200 values). 50 iterations per config, 100 ms sleep between runs.
- gpt-fast: memcpy rarely exceeds ~85% of peak HBM bandwidth. 72% MBU was considered "quite good".

## Code patterns
```python
# Graph-replay + cold-L2 timing (inference; mirrors judge rules)
flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")   # > L2, like triton do_bench
def time_graph(g, reps=200):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        flush.zero_()                     # evict weights/activations from L2 (outside the timed interval)
        s.record(); g.replay(); e.record()
        e.synchronize(); ts.append(s.elapsed_time(e) * 1e3)   # µs
    return statistics.median(ts)
t_op    = time_graph(g_op)               # graph containing only my_op(x_m)
t_empty = time_graph(g_empty)            # graph with a trivial kernel: approximates fixed graph cost
print(t_op - t_empty)
```
```bash
# Clock locking (from 3-GEMM/profile.sh). Requires root; reset afterwards.
nvidia-smi --lock-gpu-clocks=1590
nvidia-smi --lock-memory-clocks=5000
...  # runs
nvidia-smi --reset-gpu-clocks
nvidia-smi --reset-memory-clocks
```

## Pitfalls
- **Eager timing at small sizes:** kernel launch + Python dispatch is several µs, the same order as the whole kernel. Overhead-bound measurements reward fewer launches, not faster kernels (gpt-fast's overhead-bound regime).
- **Warm L2:** back-to-back reps of the same W leave it L2-resident when W fits (H200's L2 holds tens of MB, i.e. many projection matrices). The result looks faster than HBM bandwidth allows.
- **Mean vs median:** occasional preemption/clock dips skew means. The seed repo and `do_bench` both default to mean.
- **Autotune during timing:** first-call compile/tune inside the loop or graph capture.
- **Graph capture of a different code path than eager:** e.g. a size-dependent branch. Verify outputs from the replayed graph, not from eager.
- **Locked clocks ≠ judge clocks:** use locking for A/B stability, but confirm final numbers at default clocks if the judge doesn't lock. (inference)
- **Single-input reuse:** the same x every replay is fine for weights-dominated traffic, but don't let the kernel exploit it (e.g. by caching anything across calls).

## Relevance to the serving track
- The judge's rules are exactly this methodology (graph replay, cold L2, ~4 µs subtracted, no host syncs or allocation in the timed call). Reproduce them locally before trusting any speedup. Floors of 2–17 µs mean a 0.5 µs regression is 3–25%. (inference)
- Use achieved-bandwidth vs H200 peak as the primary metric for m ≤ 32. Report time vs m to verify flatness ([[speculative-decoding-shapes]]). (inference)

## Related
[[cuda-graphs]] · [[triton-docs]] · [[triton]] · [[pytorch-gpt-fast-accelerating-genai-2]] · [[speculative-decoding-shapes]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[pdl]] · [[hidden-treasures-llms]]
