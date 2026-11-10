---
title: "Yifan Yang – Using Programmatic Dependent Launch (PDL) to Reduce End-to-End Latency"
type: source
tags: [gpu, hopper, sm90, pdl, griddepcontrol, latency, launch-overhead]
source_url: https://yang-yifan.github.io/blogs/pdl/pdl.html
code_url: https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/pdl/code
---
# Yifan Yang – Using Programmatic Dependent Launch (PDL) to Reduce End-to-End Latency

Fetched in full 2026-11-08 (plus `pdl.cu` demo from the GitHub repo). Author states he made PDL happen in TensorRT-LLM's DeepSeek-R1 low-latency path.

**TL;DR**
- PDL (Hopper+) lets a **dependent kernel in the same stream** start its launch + prologue while the previous kernel is still running its mainloop and grid-ending membar.
- Two PTX instructions: `griddepcontrol.launch_dependents` (producer: "you may launch the next kernel now") and `griddepcontrol.wait` (consumer: block until the previous grid has completed and its memory is visible).
- Host side: launch the **dependent** kernel with `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization = 1`.
- `wait` placement = correctness + perf; `launch_dependents` placement = perf only. If it's missing, the trigger is implicitly at the end of the primary kernel — you still overlap launch latency + prologue with the primary's grid-ending membar.
- Overlap only happens if the SM has resources (smem, regs, threads, TMEM) for both kernels simultaneously; otherwise they serialize.

## Key ideas
- Latency anatomy of a GEMM threadblock: (1) **threadblock launch overhead** (HW), (2) **prolog** (constant loads, barrier init, TMEM alloc – independent of previous output), (3) **mainloop** (depends on previous output; loads, MMA, store), (4) **grid-ending membar** (HW; makes output globally visible).
- Without PDL: `FC1 launch + prolog + mainloop + membar + FC2 launch + prolog + mainloop + membar`. Critical path only needs `... FC1 membar + FC2 mainloop + FC2 membar`; PDL removes FC2 launch + prolog from the critical path.
- `launch_dependents` is per-CTA; the HW launches the dependent grid only after **the last CTA** of the primary issued it (or exited).
- Too late (end of kernel) → little overlap. Too early (start) → dependent prologue finishes early then just blocks in `wait`, and its prologue may **interfere with the primary's mainloop** (slowdown).
- Framework support: CUTLASS calls it Grid Dependency Control (GDC), most CUTLASS GEMMs have it; Triton via `triton.language.extra.cuda` (tutorial 11); TensorRT-LLM.
- Nsight Systems visualizes PDL overlap.
- **vs megakernel**: no-PDL = full HW sync (HW blocks launch until membar). PDL = software-assisted HW sync (HW still does membar; SW picks where to block via `wait`). Megakernel = software-only sync (SW membar + L2 atomic flag spin). Trade-off: HW sync most efficient/least flexible.
- PDL is for **dependent** kernels; independent kernels → use multi-stream.

## Numbers worth remembering
- None measured in the post (demo uses 1e9-cycle `clock64()` spin loops just to show overlap in nsys).

## Code patterns
```cpp
__global__ void pdl_kernel(...) {
  prolog(...);                                         // must not read previous kernel's output
  asm volatile("griddepcontrol.wait;");                // block until prior grid done + memory visible
  mainloop1(...);                                      // dependent work
  asm volatile("griddepcontrol.launch_dependents;");   // allow next kernel to launch
  mainloop2(...);                                      // overlaps with next kernel's prolog
}
cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attrs[0].val.programmaticStreamSerializationAllowed = 1;
cudaLaunchConfig_t cfg{}; cfg.gridDim = ...; cfg.blockDim = ...; cfg.dynamicSmemBytes = ...;
cfg.stream = s; cfg.attrs = attrs; cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, pdl_kernel, ...);
```
The demo `pdl.cu` launches the same kernel twice in one stream with the attribute on/off and times both with events.

## Pitfalls
- Removing `griddepcontrol.wait` is only legal if the kernel truly has no data dependency on the previous kernel.
- Resource oversubscription (smem/regs/threads) silently disables overlap.
- Triggering too early can slow the primary.

## Relevance to the serving track
- Per-shape floors are 2–17 µs; threadblock launch + prologue + grid-end membar are a meaningful fraction of that, and PDL is the HW mechanism that hides them in a stream of dependent kernels. (inference)
- For `out = x @ W.T`, **W does not depend on the previous kernel** – only x does. So everything W-related (mbarrier init, tensormap prefetch, TMA loads / L2 prefetch of the first W tiles) can go *before* `griddepcontrol.wait`; only the x load must come after. With a cold L2 this lets W streaming start before the producer of x finishes. (inference)
- Keep smem/regs per CTA low enough that the next kernel's CTAs can co-reside, or overlap disappears. (inference)
- Whether PDL helps in the judge depends on whether the harness launches with the PDL attribute / captures it into the graph (see [[pdl]] for the graph edge types); a kernel that contains `griddepcontrol.wait` is still correct when launched normally. (inference — the PTX doc says `wait` waits for prerequisite grids; with no programmatic dependency it simply waits for normal stream ordering, which is already satisfied.)

## Further reading
See [[yangyifan-cute-tma]] "Further reading" for the author's other Hopper posts (memory model, swizzle, CuTe copy).

## Related
[[pdl]] [[cuda-graphs]] [[tma]] [[mbarrier]] [[warp-specialization]]
