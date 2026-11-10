---
title: "Democratizing AI Compute Part 7: What about Triton and Python eDSLs? (Modular)"
type: source
tags: [gpu, triton, dsl, edsl, pallas, cutlass-python, cutile, ecosystem]
source_url: https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls
---
# Democratizing AI Compute Part 7: Triton and Python eDSLs (Modular, Chris Lattner series)

**TL;DR**
- eDSLs (Triton, Pallas, CUTLASS Python, cuTile) embed a kernel language in Python via decorators (`@triton.jit`), which capture the function, parse its AST, and compile it. They are easy for the *compiler builders* to ship, and familiar to users.
- Triton's strength is its block-centric model. One block-level `tl.dot` replaces "dozens of lines" of CUDA to drive Tensor Cores. NumPy-style broadcasting for indices, plus automatic vectorization, double buffering and software pipelining.
- Triton's cost: it is "common to lose 20%" vs expert CUDA on H100. The author says it is not well suited to max-efficiency inference. A100-tuned code often performs poorly on H100, and new architectures need restructured code to reach even ~80%. Pipelining and async memory transfers are not abstracted.
- Tooling gap: per the author, it doesn't integrate with Nsight Compute, so you are "guessing what the compiler did". The author also cites governance concerns (OpenAI owns the roadmap), lagging AMD support, and a poor fit for non-SIMT accelerators.
- Other eDSLs: Pallas (JAX/TPU, powerful, footguns), CUTLASS Python (GTC25, Blackwell-first, low level), and cuTile (GTC25, which NVIDIA reportedly admits is ~15% slower than TRT-LLM).

## Key ideas
- **Abstraction vs control** is the central trade-off. CUDA C++ gives control. eDSLs trade it for productivity.
- **eDSL failure modes:** the code "looks like Python but isn't". Generator expressions or built-in `sum` fail (Numba example), errors are opaque compiler traces, Python debuggers don't work, you can't call Python libs inside kernels, and you can't add new syntax.
- **Triton history:** Philippe Tillet's Harvard research (2019) gained momentum via OpenAI and PyTorch 2 (Inductor emits Triton).
- **Scorecard:** positive for reference implementations, developer love, and PyTorch-based scaling. Negative for top performance on new hardware, evolution speed, open community, cross-vendor performance, full programmability, and automatic fusion.
- The article notes Triton struggles with newer hardware features like TMA and FP8 (as of its writing, 2025).

## Numbers worth remembering
- ~20% typical loss vs skilled CUDA on H100 (the author's claim). Illustration: a $1B vs $800M cloud bill.
- ~80% of peak is the bar newer architectures need restructured code to reach.
- cuTile is ~15% slower than TRT-LLM (reported NVIDIA admission).

## Pitfalls
- This is an opinion piece by a competitor (Modular sells Mojo/MAX). The 20% figure is a generalization, not a benchmark. Triton has since gained TMA descriptors, warp specialization (Blackwell), Gluon, and PDL hooks ([[triton-docs]]). Treat it as directionally right for Hopper-specific tricks, possibly dated.
- The article does not discuss CuTe DSL, Helion, or ThunderKittens.

## Relevance to the serving track
- For memory-bound small-m BF16 GEMM, the "20% gap" mostly comes from features you may not need: warp specialization, wgmma pipelines, and cluster multicast matter for compute-bound tiles. A bandwidth-bound GEMV/split-K kernel can hit near-roofline in Triton. The ceiling is set by how well it keeps loads in flight. (inference)
- Where Triton hurts on these problems: fine control of register-level dequant layouts (NVFP4 MoE), TMA multicast/cluster tricks, and µs-level prologue latency. Then go to CUDA/CuTe ([[triton]] decision table). (inference)
- The tooling caveat is partly addressable. Nsight Compute does profile Triton-launched kernels at the SASS level, but source correlation is weak. Proton is Triton's own profiler. (inference)

## Related
[[triton]] · [[triton-docs]] · [[gtc25-s72449-modern-cuda]] · [[wgmma]] · [[tma]]
