---
title: PDL (Programmatic Dependent Launch)
type: concept
tags: [gpu, hopper, sm90, pdl, griddepcontrol, cuda-graphs, launch-latency]
source_url:
  - https://yang-yifan.github.io/blogs/pdl/pdl.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html   # edge data
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html   # griddepcontrol
---
# PDL (Programmatic Dependent Launch)

**TL;DR**
- CC 9.0+: a **secondary** kernel in the same stream may launch before the **primary** finishes, overlapping its CTA launch + prologue with the primary's tail + grid-end memory flush.
- Device: secondary calls `cudaGridDependencySynchronize()` = PTX `griddepcontrol.wait` before touching primary's outputs; primary optionally calls `cudaTriggerProgrammaticLaunchCompletion()` = `griddepcontrol.launch_dependents`.
- Host: launch the **secondary** with `cudaLaunchKernelEx` and `cudaLaunchAttributeProgrammaticStreamSerialization` (`programmaticStreamSerializationAllowed = 1`).
- Graphs: preserved by stream capture, or explicit edge `cudaGraphDependencyTypeProgrammatic` from port `cudaGraphKernelNodePortProgrammatic` / `...LaunchCompletion`.
- Overlap is opportunistic, needs co-resident resources, and must never be relied on for forward progress (deadlock risk).

## Key ideas
- **What overlaps** (Yang): GEMM CTA time = HW launch overhead + prolog (barrier init, constant loads, TMEM alloc; independent) + mainloop (dependent) + grid-ending membar (HW). PDL takes secondary's launch overhead + prolog off the critical path: they run concurrently with the primary's mainloop remainder + membar.
- **Trigger semantics** (PTX): `launch_dependents` lets the designated dependents be scheduled once **all CTAs** of the current grid issued it or exited; repeated calls in a CTA are no-ops; no guarantee dependents launch before completion. If never executed, trigger happens implicitly as each CTA exits (PG) — you still get overlap of launch latency/prolog with the primary's membar (Yang).
- **Wait semantics** (PTX): `griddepcontrol.wait` blocks until all prerequisite grids in flight have **completed and their memory operations are performed and visible** to the current grid. Note: wait is on full completion, not on the trigger. "If the prerequisite grid uses launch_dependents, the dependent must use griddepcontrol.wait."
- **Memory ordering of the trigger** (PTX): a release fence before `launch_dependents` synchronizes with the *start* of the dependent grid if same memory-sync domain + fence scope `gpu` (or scope `sys`). Basis for consuming partial data before `wait`.
- **Fine-grained variant** (PG): secondary may poll sentinel values in produced data to consume parts early, but must still call `cudaGridDependencySynchronize`.
- **Placement**: `wait` → correctness + perf; `launch_dependents` → perf only. Too early: secondary's prolog finishes then idles in `wait` and can steal resources from primary's mainloop. Too late: little overlap.
- **Requirements** (Yang §2.1): correct `wait`; secondary launched with the attribute; SM has smem/regs/threads/warps/TMEM for both CTAs at once (else serial); trigger in primary optional.
- **vs megakernel**: PDL = HW membar + SW-placed wait; megakernel = SW fences + L2 flag spin (more flexible, less efficient). See memory-model patterns in [[mbarrier]].
- **Ecosystem**: CUTLASS "GDC" (`cutlass/arch/grid_dependency_control.h`, most GEMMs); Triton `triton.language.extra.cuda` (tutorial 11); TensorRT-LLM DeepSeek-R1 low-latency.

## CUDA Graph interaction
- Edge data (CUDA 12.3+): the only non-default edge type is `cudaGraphDependencyTypeProgrammatic`, between two kernel nodes; it makes the upstream kernel visible to a downstream `cudaGridDependencySynchronize()`. Must use outgoing port `cudaGraphKernelNodePortProgrammatic` or `cudaGraphKernelNodePortLaunchCompletion`.
- Stream-capture equivalents (PG table):
  - `cudaLaunchAttributeProgrammaticStreamSerialization = 1` → edge type Programmatic, from_port `cudaGraphKernelNodePortProgrammatic`.
  - `cudaLaunchAttributeProgrammaticEvent` with `triggerAtBlockStart = 0` → same (Programmatic port).
  - `cudaLaunchAttributeProgrammaticEvent` with `triggerAtBlockStart = 1` → from_port `cudaGraphKernelNodePortLaunchCompletion`.
- Programmatic edges don't wait for full upstream completion, so they're ignored when checking capture rejoin and can't be discarded at capture end (graphs guide).

## Code patterns
```cpp
__global__ void k(const __grid_constant__ CUtensorMap tmW, const bf16* x, bf16* out) {
  // ---- prologue: independent of previous kernel ----
  init_mbarriers(); prefetch_tensormap(&tmW);
  issue_tma_W_stages();                 // W is a constant weight: legal before wait (inference)
  asm volatile("griddepcontrol.wait;" ::: "memory");   // == cudaGridDependencySynchronize()
  // ---- dependent: x was produced by previous kernel ----
  load_x(); mainloop(); 
  asm volatile("griddepcontrol.launch_dependents;");   // == cudaTriggerProgrammaticLaunchCompletion()
  epilogue_store(out);
}
cudaLaunchAttribute a[1];
a[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
a[0].val.programmaticStreamSerializationAllowed = 1;
cudaLaunchConfig_t cfg{}; cfg.gridDim = g; cfg.blockDim = b; cfg.stream = s; cfg.attrs = a; cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, k, tmW, x, out);
```
Triton (tutorial 11, checked): in-kernel `tl.extra.cuda.gdc_wait()` / `tl.extra.cuda.gdc_launch_dependents()` (the latter "by itself provides no additional memory-ordering guarantees, unlike gdc_wait"); launch with `kernel[grid](..., launch_pdl=True)`.

## Pitfalls
- Reading anything written by the previous kernel (x, or `out` if it was being written/zeroed earlier) before `wait` = race.
- **Write-after-read hazard (inference)**: if your kernel *writes* a buffer the previous kernel still reads, that write must also come after `wait`.
- Large smem/reg footprint → no co-residency → no overlap; persistent kernels occupying all SMs block the secondary's CTAs until they exit. (inference)
- The attribute goes on the **dependent** launch; setting it on the primary alone does nothing for the primary.
- Not supported pre-Hopper: APIs require CC 9.0 for overlap.

## Relevance to the serving track
- Floors of 2–17 µs per shape make launch + prologue + grid-end flush a large fraction; PDL is the only non-megakernel way to hide them between dependent kernels in a graph. (inference)
- **Key trick for `x @ W.T`**: W is a static weight, so mbarrier init, `prefetch.tensormap`, and TMA loads / `cp.async.bulk.prefetch.tensor.L2` of W can all precede `griddepcontrol.wait`; only x (and the output write) must follow. Under a cold L2 this starts HBM streaming of W during the previous kernel's tail. (inference)
- Whether this pays off in the judge depends on whether the harness launches/captures your kernel with the programmatic attribute after a preceding kernel; if it doesn't, `griddepcontrol.wait` costs ~nothing (prerequisites already complete) so it's safe to include. Verify against SGLang's kernels and the harness. (inference)
- Call `launch_dependents` as early as is safe (e.g. right after the last W load is issued) so the *next* layer's prologue overlaps your epilogue. (inference)

## Related
[[yangyifan-pdl]] [[cuda-graphs]] [[tma]] [[mbarrier]] [[warp-specialization]]
