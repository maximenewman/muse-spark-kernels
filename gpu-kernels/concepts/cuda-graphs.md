---
title: CUDA Graphs
type: concept
tags: [gpu, cuda, cuda-graphs, launch-overhead, pdl, benchmarking, serving]
source_url:
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
  - https://docs.pytorch.org/docs/2.15/notes/cuda.html
  - https://pytorch.org/blog/accelerating-generative-ai-2/
  - https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
---
# CUDA Graphs

**TL;DR**
- A graph is a DAG of work (kernel, memcpy, memset, host, event, child-graph, conditional, and mem-alloc/free nodes). There are three stages: **define → instantiate (`cudaGraphExec_t`) → launch**. Per-kernel driver setup is paid once at instantiate, which matters most for short kernels.
- Two ways to build one: explicit `cudaGraphAddNode` or **stream capture** (`cudaStreamBeginCapture` … `cudaStreamEndCapture`). PyTorch wraps capture as `torch.cuda.graph` / `CUDAGraph.replay()`. SGLang and the competition harness capture each shape this way.
- Capture freezes **kernel pointers, launch dims, and argument values, including tensor addresses**. Replays reuse the same virtual addresses, so inputs/outputs must be **static buffers** you copy into. Shapes and branching are frozen at capture time.
- Illegal during capture: syncing or querying the capturing stream/event/device, using the legacy NULL stream (unless other streams are `cudaStreamNonBlocking`), synchronous `cudaMemcpy`, and CPU reads like `.item()`. Any illegal op invalidates the capture, and `EndCapture` returns an error and a NULL graph.
- PDL survives capture: a launch with `cudaLaunchAttributeProgrammaticStreamSerialization` becomes a `cudaGraphDependencyTypeProgrammatic` edge from `cudaGraphKernelNodePortProgrammatic`. So [[pdl]] overlap works inside replayed graphs.

## Key ideas
**Why graphs.** gpt-fast went 25.5 → 107 tok/s mostly by removing CPU launch overhead (`mode="reduce-overhead"` = CUDA graphs + static KV cache). See [[pytorch-gpt-fast-accelerating-genai-2]]. Replay skips Python, C++ dispatcher, and driver argument setup (PyTorch notes).

**Stream capture rules (CUDA guide).**
- Any stream except the legacy NULL stream. PyTorch requires a non-default stream for raw `capture_begin`/`capture_end`. `torch.cuda.graph` picks a side stream for you.
- Cross-stream fork/join via `cudaEventRecord` + `cudaStreamWaitEvent` with events recorded *in the same capture*. All forked streams must rejoin the origin stream before `cudaStreamEndCapture` (on the origin stream).
- Waiting on a non-captured event requires `cudaEventWaitExternal`. Merging two capture graphs is invalid.
- `cudaMallocAsync`/`cudaFreeAsync` captured into a graph become **memory nodes** with fixed addresses across launches. `cudaGraphInstantiateFlagAutoFreeOnLaunch` frees unfreed allocations on relaunch.

**What breaks capture or silently breaks replay (PyTorch notes).**
- Hard errors: CPU-GPU sync (`.item()`, `.cpu()`, `nonzero()` with dynamic output, printing tensors), and dynamic control flow (except `torch.cond`, which becomes conditional nodes under `backend="cudagraphs"`).
- Silent UB: dynamic shapes, other non-captured CUDA work on any thread during capture, and depending on CPU-side work (it is **elided on replay**: Python code inside the `with` block runs once, at capture). Only one capture per process at a time.
- RNG: allowed, but use `graphsafe_get_state`/`graphsafe_set_state` with multiple generators.
- Allocations: during capture, PyTorch allocates from a **private memory pool** owned by the graph. Those addresses stay reserved, so outputs are overwritten on the next replay unless you `.clone()` them. Graphs can share a pool (`pool=g1.pool()` or a `MemPool`) only if they always replay in capture order and never concurrently.
- Warm up eagerly on a side stream (≥ a few iterations) *before* capture. This triggers lazy init: cuBLAS workspace, Triton JIT/autotune, NCCL comms.
- A captured graph can be replayed on any stream.

**Updating without re-capture.** `cudaGraphExecUpdate` takes a topologically identical graph. Per-node updates use `cudaGraphExecKernelNodeSetParams` etc., and `cudaGraphNodeSetEnabled` toggles kernel/memcpy/memset nodes. Topology changes need re-instantiation. A `cudaGraphExec_t` never runs concurrently with itself, so launches of the same exec are serialized.

**Conditional nodes** (IF/else, WHILE, SWITCH; handle via `cudaGraphConditionalHandleCreate`; device sets it with `cudaGraphSetConditional`). This is the only legal way to get data-dependent control flow inside a graph. Bodies may contain only kernel/empty/memcpy/memset/child/conditional nodes.

**Device graph launch:** `cudaGraphInstantiateFlagDeviceLaunch` with fire-and-forget / tail launch streams (≤120 fire-and-forget per parent launch, ≤255 pending tail launches).

**PDL edges in graphs** (CUDA guide, PDL section).
| Stream-side launch attribute | Captured edge type | from_port |
|---|---|---|
| `cudaLaunchAttributeProgrammaticStreamSerialization`, `programmaticStreamSerializationAllowed=1` | `cudaGraphDependencyTypeProgrammatic` | `cudaGraphKernelNodePortProgrammatic` |
| `cudaLaunchAttributeProgrammaticEvent`, `triggerAtBlockStart=0` | same | `cudaGraphKernelNodePortProgrammatic` |
| `cudaLaunchAttributeProgrammaticEvent`, `triggerAtBlockStart=1` | same | `cudaGraphKernelNodePortLaunchCompletion` |
- Edge data (out-port, in-port, type) can also be set explicitly via `cudaStreamBeginCaptureToGraph` / `cudaStreamUpdateCaptureDependencies`. Only kernel nodes have extra out-ports.
- Semantics: the secondary may launch once all primary CTAs have called `cudaTriggerProgrammaticLaunchCompletion` (or exited). The trigger gives **no visibility** of primary writes. The secondary must call `cudaGridDependencySynchronize()` before consuming them, which waits for completion *and flush to global memory*. Overlap is opportunistic. Relying on concurrency can deadlock. In Triton the equivalents are `gdc_launch_dependents` / `gdc_wait` ([[triton-docs]]).

## Numbers worth remembering
- gpt-fast: 25.5 → 107.0 tok/s (Llama-7B, A100) from compile + reduce-overhead (CUDA graphs) + static KV cache.
- Device graph launch limits: 120 fire-and-forget per launch, 255 pending tail launches.
- Harness: ~4 µs graph fixed cost subtracted per replay (competition rules, not a CUDA-doc number).

## Code patterns
```python
# PyTorch manual capture (from the PyTorch CUDA notes)
g = torch.cuda.CUDAGraph()
static_in = torch.empty((5,), device="cuda")            # static input buffer: fixed address
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):                               # warmup on side stream: lazy init, JIT, autotune
    for _ in range(3): static_out = static_in * 2
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):                                # capture; allocations come from g's private pool
    static_out = static_in * 2
static_in.copy_(torch.full((5,), 3, device="cuda"))     # feed new data by copying INTO the static buffer
g.replay()                                               # static_out now holds 6s
```
```cpp
// Raw CUDA capture of a kernel launched with PDL -> programmatic edge in the graph
cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
producer<<<g1, b1, 0, s>>>(...);
cudaLaunchAttribute a{}; a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
a.val.programmaticStreamSerializationAllowed = 1;
cudaLaunchConfig_t cfg{g2, b2, 0, s, &a, 1};
cudaLaunchKernelEx(&cfg, consumer, ...);                 // consumer calls cudaGridDependencySynchronize()
cudaStreamEndCapture(s, &graph);
cudaGraphInstantiate(&exec, graph, 0);
cudaGraphLaunch(exec, s);
```

## Pitfalls
- **Allocating inside the timed/captured call.** `torch.empty` inside capture works (private pool) but returns the *same* address every replay. Never allocate per call in a serving kernel wrapper. Pre-allocate outputs/workspaces (split-K partials, semaphores) in `prepare()`. (inference, plus the PyTorch pool semantics above)
- **Branching on input values** (e.g. `if x.abs().max() > …`) forces a sync and fails capture. Branching on *shape* is fine, because it is resolved at capture, so the captured path is per-shape. (inference)
- **Split-K semaphores/counters** must be reset *inside* the graph (a memset node or the kernel itself resets them). A host-side reset before the first capture runs only once. (inference)
- **Autotuners/JIT** (Triton, torch.compile) must have run for every captured shape before capture. A first-call benchmark inside capture breaks or bakes in garbage.
- Async-to-host copies and host callbacks are elided or frozen. Python side effects happen once.
- Graph replay timing includes inter-node launch gaps but excludes CPU overhead. Eager timing of a µs kernel measures the CPU. See [[benchmarking-kernels]].

## Relevance to the serving track
- The harness captures each m ∈ {1,8,16,32,128,256} separately. You can dispatch to a different kernel/config per m in Python at capture time with zero runtime cost. Do it. (inference)
- Since the ~4 µs fixed cost is subtracted, what's left is kernel device time plus any extra nodes you add (memsets, reduction kernels). A split-K that needs a second reduction kernel or a memset node pays an extra node's launch latency inside the graph. Prefer a single kernel with an in-kernel reduction (atomics into fp32, or a last-CTA reduction with a self-resetting counter). (inference)
- PDL edges are preserved by capture. In SGLang's real graph your GEMM could overlap its prologue (weight prefetch is independent of x!) with the previous kernel's tail. Weight loads don't depend on the producer, so issue W loads before `cudaGridDependencySynchronize()`/`gdc_wait` and only wait before reading x. Whether the harness rewards it depends on what precedes your kernel in the timed graph. See [[pdl]]. (inference)
- Cold L2 is enforced by the harness, so weights come from HBM every replay. Don't rely on L2 residency between replays. See [[benchmarking-kernels]]. (inference)

## Related
[[pdl]] · [[benchmarking-kernels]] · [[pytorch-gpt-fast-accelerating-genai-2]] · [[triton]] · [[triton-docs]] · [[split-k-and-stream-k]] · [[gemv-and-skinny-gemm]] · [[speculative-decoding-shapes]]
