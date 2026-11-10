---
title: Serving track playbook (Muse Spark Open Kernel Challenge)
type: playbook
tags: [muse, challenge, serving, sglang, strategy]
source_url:
  - raw/okc/llms.txt         # https://openkernelchallenge.dev/llms.txt, fetched 2026-11-08
  - raw/okc/serving.json     # GET https://openkernelchallenge.dev/api/v1/serving
---
# Serving track playbook

**TL;DR**
- A submission is a **git diff against SGLang commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`**, touching only `python/sglang/` and only source/JSON files. The judge runs it on a **whole 8×H200 node** serving `Muse-Spark-Open-816B-A42B-NVFP4` with the DFlash drafter.
- Score: **throughput at stock SGLang's interactivity, vs stock on the same node**, over a ladder of concurrency 1 / 4 / 16 / 32. TTFT may be at most 2× stock's at each load. Outputs must match stock's about as well as stock matches itself, and every request returns exactly 220 tokens.
- Opens **2026-11-10 09:00 PT**, closes **2026-11-19 23:59 PT**. You get **1 scored run and 3 checks per day** (reset 00:00 UTC), each run up to 90 minutes. The leaderboard stays private until the end, and **SGLang maintainers' judgment of whether a patch could be merged factors into the winners.** Prizes total $100k ($50k for first).
- Where to spend effort (inference from the kernel-track baselines): MoE (Marlin NVFP4) ≈ 60% of a decode step > dense projections ≈ 21% > attention ≈ 17%. Small-m decode/verify dominates.

## Workflow
```bash
okc serve init [dir]                 # pinned SGLang + patch rules + empty patch.diff
# edit serving/sglang/python/sglang/...
okc serve diff                       # writes serving/patch.diff
okc serve check serving/patch.diff   # unscored; one rung: concurrency 16, 1 rep, 35 min deadline
okc serve submit serving/patch.diff  # scored; whole ladder
okc serve status [job]               # your runs and scores (board is private)
okc withdraw <job>
```
HTTP: `GET /api/v1/serving` (rules, ladder), `POST /api/v1/serving/submissions {"patch": ..., "mode": "run"|"check"}`. See `raw/okc/llms.txt`.

Runs execute inside the `lmsysorg/sglang:dev-muse-spark-open` image (pinned by digest in `serving.json`), which has PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS. There's no network access. **The prebuilt `sgl-kernel` binaries aren't rebuilt**, so new CUDA has to be JIT-built from source inside `python/sglang` (Triton, `load_inline`/NVRTC, CuTe DSL, TileLang). Compile time counts toward the 90 minutes but not toward the score.

## The ladder (from serving.json)
| concurrency | reps | conversations/slot |
|---|---|---|
| 1 | 3 | 2 |
| 4 | 3 | 2 |
| 16 | 3 | 2 |
| 32 | 1 | 1 |
Check runs use only concurrency 16 (1 rep), re-scoring 8 conversations. The workload is **agentic coding turns**: ~80K-token contexts, 753-token new turns, 16K first-turn chunks (from the prefill problem). TTFT is capped, so prefill can't regress more than 2×.

## Stock launch config (what you're beating)
```
--model-path meta-models/Muse-Spark-Open-816B-A42B-NVFP4
--speculative-draft-model-path RadixArk/Muse-Spark-Open-DFlash2
--speculative-algorithm DFLASH --speculative-num-draft-tokens 8
--tp-size 8 --ep-size 8 --quantization modelopt_fp4 --moe-runner-backend marlin
--mem-fraction-static 0.85 --enable-hierarchical-cache --chunked-prefill-size 16384
--context-length 131072 --max-running-requests 32 --cuda-graph-max-bs-decode 32
--enable-cache-report --incremental-streaming-output --random-seed 1234
env: SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_TIMEOUT_KEEP_ALIVE=1800
```
You can't change the launch args (the judge launches); you change the code they select. JSON configs under `python/sglang` are allowed, e.g. kernel tuning tables (inference).

## Hard rules (paraphrased; exact text in serving.json / llms.txt)
- Don't touch `python/sglang/srt/model_loader/`. No renames, mode changes or binary patches. At most 400 KB, no added line over 400 characters. Forbidden words include `KC_RESULT`, `SIMULATE` and `kc-secret` (full list in serving.json).
- The model stays the model: 8 routed experts per token, the 2,048-token window, every norm, and the parameter names and shapes SGLang registers (repacking after load is OK).
- No precision below stock: BF16 activations stay BF16 where stock keeps them, the FP8 KV cache stays FP8, and stock's FP8 query in FA3 stays as it is.
- The drafter is DFlash with its released weights. Making it faster is allowed; changing, retraining or replacing it is not. Extra drafting may only use the requests themselves (prompt lookup, n-grams).
- Readable source only. **A patch that special-cases the workload or its load levels, breaks other models or GPUs, or is too sprawling to review can be disqualified.** Write it like an upstreamable PR.

## Per-GPU decode-step budget (stock, from the kernel-track problems)
| Component | SGLang time per decode step | Kernel-track problem |
|---|---|---|
| Routed experts (NVFP4, Marlin) | 13.45 ms | `muse_moe_w4a16` |
| Dense projections (qkv_gate, o, pooler, unpooler, router) | 4.83 ms | `muse_bf16_proj` |
| Attention (FA3, FP8 paged KV, 80K ctx) | 3.77 ms | `muse_attention_decode` |
| Whole layer (incl. norms/residuals) | 22.62 ms | `muse_decode_layer` |
These are the kernel judge's load-averaged sums over 62 layers. Use them as relative weights, not as wall-clock (inference). Missing from them: all-reduce/NCCL (2 per layer under TP8), the DFlash drafter's forward, sampling/verification, scheduler/host overhead, and prefill.

## Lever map (inference: candidate ideas, ranked by expected payoff; check each against the rules)
1. **MoE W4A16 kernel at m ≤ 256 tokens × top-8 routing over 32 local experts.** This is a weight-streaming grouped GEMM: each active expert's NVFP4 weights (`2I×H/2` + scales) have to be read from HBM. Wins come from reading each expert once per step for all its tokens, fusing SwiGLU + post-expert RMSNorm + routing weight into the epilogue, good SM load balance under skewed routing, and in-register FP4→BF16 dequant. → [[weight-only-quantization]], [[gemv-and-skinny-gemm]], [[split-k-and-stream-k]], [[persistent-kernels-and-tile-scheduling]], [[tma]]
2. **Dense BF16 projections**: split-K / swap-AB skinny GEMMs that stream W at full HBM bandwidth. The user's kernel-track work in `../muse_bf16_proj/submission.py` (split-K Triton, swap-AB, atomic-ticket reduction, CUDA-core GEMV for m=1) is directly portable. → [[gemv-and-skinny-gemm]], [[memory-bandwidth-and-latency]]
3. **Fusion across ops**: norms + residual + quant/dequant + router top-k fused into neighbouring GEMM epilogues/prologues, fewer launches per layer. Use PDL to overlap each kernel's prologue with the previous kernel's tail inside the CUDA graph. → [[pdl]], [[cuda-graphs]]
4. **Attention decode/verify**: FA3 with FP8 KV at page_size 1. Split-KV for the 80K full layers at small batch; the sliding layers only read 2K keys. → [[split-k-and-stream-k]]
5. **Communication**: two all-reduces per layer at small messages are latency-bound. Fusing all-reduce with the residual and norm is a known SGLang/FlashInfer pattern (inference: check what stock already does at `a55ab7d5`).
6. **Spec decoding**: DFlash drafter speed (it's an allowed target), plus prompt-lookup / n-gram drafting on agentic-coding contexts, which repeat a lot of text from the conversation.
7. **Scheduler/host overhead** at concurrency 1–4, where GPU time per step is small.

## Measuring locally
The user has H100s, not H200s, and not a full 8-GPU serving setup. Single H100-80GB nodes are available on Alliance Rorqual (`ssh rorqual`; MFA, so the user opens the master socket). See [[benchmarking-kernels]]. Per-kernel speedups can be validated on one GPU with the kernel-track harnesses (`okc init <problem>` gives reference.py/baseline.py). End-to-end throughput only shows up in `okc serve check` (3/day).

## Related
[[muse-spark-open-model]] · [[speculative-decoding-shapes]] · [[gemv-and-skinny-gemm]] · [[cuda-graphs]] · [[pdl]] · [[weight-only-quantization]] · [[triton]]
