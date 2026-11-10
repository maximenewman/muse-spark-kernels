---
title: "Split-KV (flash-decoding) attention over a paged FP8 cache, in Triton"
type: concept
tags: [gpu, attention, decode, flash-decoding, split-kv, gqa, paged-kv, fp8, triton, muse]
source_url:
  - ../muse_attention_decode/submission.py      # the implementation measured here
  - ../muse_attention_decode/TASK.md
  - raw/okc/problem-muse_attention_decode.txt
---
# Split-KV decode attention over a paged FP8 cache

**TL;DR**
- Decode/verify attention at 80K context is **HBM-bound for L=1 and close to compute-bound for L=8**.
- For `muse_attention_decode`, a two-kernel Triton design beat SGLang's FA3 pipeline **1.44× on H100**. The design:
  - a prep kernel: q/k RMSNorm, RoPE, FP8 cache write
  - one split-KV attention kernel, whose last split per (request, head group) merges the partials and applies `rmsnorm × sigmoid(gate)`
- The biggest wins, in order:
  1. A **vectorized split merge** (the serial merge loop was L2-latency-bound).
  2. **Enough CTAs** at batch 1 (≥ 264).
  3. **FP8 QK** (q rounded to E4M3, as FA3 does).
  4. **One CTA reads both KV heads' 128 B cache row** through a block-diagonal Q.
- **Still losing:** verify (L=8) full-context layers, 0.79–0.93× FA3. That regime needs FA3-class wgmma + `exp2` throughput, not just more bandwidth.

## Key ideas
- **Shape facts:** Muse TP8 has 16 q heads and 2 KV heads (GQA 8) with head dim 64. The FP8 cache with page size 1 stores one token's K as a contiguous **128 B row (both KV heads)**, at a random location. Verify blocks have L=8, causal inside the block.
- **Grid from shapes only** (CUDA-graph safe):
  - `nsplit = ceil(span / chunk)`, where span = window + L − 1 on sliding layers and page-table capacity on full ones.
  - Splits past a request's length do no work. Every split takes a ticket.
- **Block-diagonal GQA trick:** with KVPC = 2 KV heads per CTA, Q is `[ROWS, 128]`, and the rows for KV head 0 are zero in columns 64–127 (and vice versa). One `tl.dot(q, kᵀ)` against the full 128 B row then gives each row its own head's scores. P·V produces both halves, and each row keeps its own-head half on store.
  - Measured it on decode L=1, which has 16 rows and no padding. For L=8 (64–128 rows), one KV head per CTA was faster.
- **Self-resetting ticket merge** ([[split-k-and-stream-k]]):
  - Each split writes (m, l, acc), with acc holding only the row's own-head D columns.
  - The last split loads every (m, l) as one `[MAXS, ROWS]` tile, then accumulates acc in blocks of SB splits (`tl.range`, `num_stages=3`).
  - The **serial `for s in range(nsplit)` version was the bottleneck at batch 1**: 126 dependent L2 round-trips.
- **Masks only on boundary blocks:** a block needs no mask if `t0 ≥ seq + L − W` (window) and `t0 + BN ≤ seq + 1` (causal). Every other block skips the `where` work.
- **FP8 paths:**
  - **QK:** q is stored as E4M3 of the normalized query; the 5.473 factor moves into the softmax scale. That removes K's FP8→BF16 conversion, which the profile showed loading the XU pipe heavily.
  - **P·V in FP8** (P rounded to E4M3): helped decode-full shapes (2–3%) but **hurt verify a lot** (468 → 598 µs at B=32), so it's per-shape.

## Numbers worth remembering (H100, cold L2, graph replay, µs; SGLang = FA3 pipeline)
| shape | SGLang | ours | |
|---|---|---|---|
| decode 1 sliding | 22.3 | 8.4 | 2.67× |
| decode 1 full (80K) | 33.3 | 23.7 | 1.41× |
| decode 32 full | 543 | 274 | 1.99× (662 MB → 2.4 TB/s) |
| verify 32 sliding | 36.6 | 24.9 | 1.47× |
| verify 32 full | 425 | 467 | 0.91× |
| **geomean of the 6 loads** | | | **1.44×** |

- SGLang's sliding layers cost 22–37 µs for 0.1–3.7 µs of data, because 9 small kernels each take 3–6 µs.
- Fusing the norm-gate and the combine into the attention kernel is most of the win on those shapes.

## Pitfalls (all found with ncu, [[findings-log]] 2026-11-09)
- **Per-head loops in a tiny kernel:** `tl.static_range(16)` over heads, each with its own load → reduce → store, ran at 12.5 µs for one token (long_scoreboard 18). Loading all heads as one `[HQ, D/2]` tile cut it to 3.3 µs.
- **Too few CTAs:** a 512-token minimum chunk gave 4 CTAs for batch-1 sliding (22 µs). Size splits so the grid holds ≥ 2 CTAs per SM; tuning picked 264–1056.
- **Serial merge loops:** each iteration is an L2 round-trip, and Triton doesn't overlap them. Vectorize or block-pipeline them.
- Reading 64 B per token (one head per CTA) from 128 B random rows wastes half of each row's fetch.

## Relevance to the serving track
- SGLang's Muse decode attention pays ~5 µs per launch over 9 kernels. A fused prep + split-KV + merge/norm-gate path is an upstreamable win for sliding layers and decode-full layers.
- Verify (spec-decode) full layers need a better kernel than Triton's: wgmma, FP8 QK and possibly polynomial `exp2` offload (FA3/FA4 style).

## Related
[[split-k-and-stream-k]] · [[memory-bandwidth-and-latency]] · [[triton]] · [[speculative-decoding-shapes]] · [[cuda-graphs]] · [[findings-log]] · [[rorqual-h100-workflow]] · [[muse-spark-open-model]]
