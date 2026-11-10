---
title: Findings log
type: playbook
tags: [muse, experiments, log]
---
# Findings log

An append-only log of measured results. Newest first. Each entry gives the date, GPU (H100 Rorqual / H200 judge), problem/shape, the change, the measured numbers, the command, and a verdict (kept / reverted / why).

## 2026-11-10 · H100 (Rorqual) · muse_moe_w4a16: 1.30× → 1.60×
Commands (in `../muse_moe_w4a16`): `sbatch job.sh python bench.py`, `python tune.py --m ... --tb ... --nst ... --slots ... --ks ... --w ...`, `DISABLE_DCGM=1 sbatch prof_job.sh`. Design: [[nvfp4-moe-weight-streaming]].

| step | geomean | change | evidence |
|---|---|---|---|
| v1 | 1.30 | route + 2 mma.sync GEMMs (FP4→BF16 in registers, bulk-copy ring) + norm-sum | correct first try (3.2e-3 vs 1.5% bound) |
| v1b | 0.53 | TB 32/64 tokens per item to stop re-streaming hot experts | reverted: 130–172 KB smem → 1 CTA/SM; ncu issue-latency-bound |
| v2 | 1.35 | shift+mask FP4 layout (11 vs 15 ops / 8 weights), own NVRTC/ctypes launcher | torch launcher spawned gcc per launch |
| v3 | 1.55 | tuned (TB, stages, slots): 2–4 stages win (occupancy) | verify32 607 → 445 µs |
| v4 | **1.60** | unconditional first token block (no predicated HMMA + WARPSYNC) | 2–4% on every load |

- **No gain (kept as tunables):** split-K (1/2/4) at small M; 2- and 8-warp CTAs; item-major grid order for L2 reuse.
- **ncu at batch 1:** a copy-only variant still takes 17 + 12 µs (2.2 / 1.6 TB/s). At 256 tokens: issue 69%, about 43 instructions per weight k-step (hot experts re-dequantized per item).

## 2026-11-09 · H100 (Rorqual) · muse_attention_decode: from 0.68× to 1.44×
Commands (in `../muse_attention_decode`):
- `sbatch job.sh python bench.py` (16 tests + 12 scored shapes against the real SGLang/FA3 baseline)
- `python tune.py --layer sliding|full`
- `DISABLE_DCGM=1 sbatch prof_job.sh`

Design: [[split-kv-decode-attention]]. Baseline setup: [[rorqual-h100-workflow]].

| step | geomean | what changed | evidence |
|---|---|---|---|
| v1 | 0.68 | prep kernel + per-(request, KV head) split-KV Triton kernel + ticket merge | correct (errors ~1e-3 against 3–15% bounds) |
| v2 | 0.64 | both KV heads per CTA (block-diagonal Q), boundary-only masks | decode32 full 541 → 406 µs; verify worse |
| v3 | 0.73 (1.06 tuned) | ncu: prep kernel took 12.5 µs (serial per-head chain), grid only 4 CTAs at B=1 | vectorized prep → 3.3 µs; tunable splits |
| v4 | 1.11 | vectorized split merge (`[MAXS, ROWS]` m/l tile + pipelined acc blocks) | decode1 full 46 → 33 µs, decode1 sliding 11 → 8.8 µs |
| v5 | 1.42 | FP8 QK (q → E4M3) + re-tune | decode32 full 333 → 282 µs; every shape faster |
| v6 | **1.44** | FP8 P·V only on decode-full + decode1-sliding | P·V FP8 hurt verify (468 → 598 µs) |

- Still losing: verify full (L=8) at 0.79–0.93×.
- ncu for verify32 full (v3): XU pipe 54.5%, tensor pipe 17.8%, 150 M instructions. Next steps there: wgmma-friendly tiles, fewer conversions, or polynomial `exp2`.

**Carried back to muse_bf16_proj:** unrolled the split-K merge loops (Triton: 4 independent loads per step; CUDA: `#pragma unroll 8`).
- Result: **1.169 / 1.172×, no change.** Its split counts are ≤ 16 with small tiles, so the serial merge was cheap there.
- The attention lesson (a vectorized merge mattered) applies when splits × tile size is large.

## 2026-11-09 · H100 (Rorqual) · muse_bf16_proj: Nsight Compute profiling of the CUDA kernels
Commands (in `../muse_bf16_proj`):
- `DISABLE_DCGM=1 sbatch prof_job.sh` (runs `ncu --profile-from-start off --set full`)
- `python ncu_summary.py ncu/*.ncu-rep`

ncu locks clocks to base, so absolute times are about 1.3× the bench times. Compare kernels only against each other. See [[rorqual-h100-workflow]].

**1. Why register-direct CUDA lost** (qkv_gate):

| kernel | µs (ncu) | DRAM TB/s | warps/sched | eligible | top stall (cyc/issue) | L1 ld sectors |
|---|---|---|---|---|---|---|
| m=1 Triton gemv (576×128 thr, 70 regs) | 17.2 | 2.31 | 4.16 | 0.25 | long_scoreboard 17.7 | 1.47 M |
| m=1 CUDA v2 (144×256, 99 regs, 2 CTA/SM) | 19.5 | 2.04 | 2.11 | 0.10 | long_scoreboard 18.3 | **2.51 M** |
| m=1 CUDA v3 | 22.3 | 1.78 | 2.30 | 0.08 | long_scoreboard 22.1 | 1.25 M |
| m=8 Triton gemm (72×128, 5-stage ring, 102 KB smem) | 18.9 | 2.13 | **0.99** | 0.15 | long_scoreboard **2.5** | — |
| m=8 CUDA v2 | 21.0 | — | 2.16 | 0.09 | long_scoreboard 20.2 | 3.54 M |
| m=8 CUDA v3 | 27.7 | — | 2.21 | 0.07 | long_scoreboard 29.0 | 1.77 M |

- v2 really did request each sector twice (2.51 M against the ideal ~1.2 M). v3 fixed that and got *slower*, so sector efficiency wasn't the bottleneck.
- The bottleneck is **bytes in flight**. Register-held W caps occupancy (99 regs → 2 CTAs/SM), and warps stall on `long_scoreboard`. Triton's gemm reaches higher bandwidth with **one warp per scheduler**, because its in-flight bytes sit in a 5-stage `cp.async` shared-memory ring.

**2. CUDA v4: W through a `cp.async` ring** (16 B chunks XOR-swizzled by `row & 7`, 0 bank conflicts) — no better.
- First try: qkv m=8 took **57 µs** (0.71 TB/s), the same at 3 and 6 stages.
- ncu showed 3.3 M instructions against Triton's 1.2 M, ~3,000 cycles per 128-k iteration against Triton's ~1,050, and low stall-per-issue.
- Cause: **x still came from global memory every 64 k**. ~96 KB of smem per CTA leaves too little L1 for x (64 KB per CTA slice), so each step waited on an L2 round-trip, with nothing to hide it at 1 warp per scheduler.

**3. v4 + x staged through the same ring** (`cp.async` with src-size 0 to zero-fill tokens ≥ m). This brought qkv m=8 from 28.8 to **15.8 µs** (bench timing) and pooler m=8 from 28.8 to 25.1 µs.
- That's within 5–10% of Triton at m=1–16, but it doesn't win any shape. At m=32 it's still 1.3–1.5× slower.

**Kept:** the submission is unchanged (1.17×): Triton everywhere, plus register-direct CUDA for router m=8.

**Lessons:**
- At m ≥ 8, a skinny GEMM needs its in-flight bytes in smem through async copies, not registers.
- **Every** operand of the inner loop must be pipelined, x included. One unpipelined small load serializes the whole loop when occupancy is one warp per scheduler.
- Don't trust a sector-efficiency theory without measuring warps, eligibility and stalls.

## 2026-11-08 (later) · H100 (Rorqual) · muse_bf16_proj: vault-guided round
Commands: `sbatch job.sh python tune.py [--cuda-only] --m <ms>` and `sbatch job.sh python bench.py` in `../muse_bf16_proj` ([[rorqual-h100-workflow]]).

**Result: 1.171× / 1.169× geomean** on two runs (was 1.15×).

| load | before | after |
|---|---|---|
| decode 1 | 1.25 | 1.25 |
| decode 16 | 1.15 | 1.15 |
| decode 32 | 1.16 | 1.16 |
| verify 1 | 1.39 | 1.39 |
| verify 16 | 1.06 | **1.08** |
| verify 32 | 0.97 | **1.04** |

What was tried, guided by [[gemv-and-skinny-gemm]] (options iii and 6) and [[wgmma]]:
1. **Hand-written CUDA, register-direct W stream into mma.sync fragments** ([[register-direct-weight-streaming]]). Correct on 2,688/2,688 configs, but **slower than tuned Triton on every BF16 shape**:
   - qkv_gate m=1: 15.9 vs 12.9 µs
   - qkv_gate m=8: 17.3 vs 14.9 µs
   - m=32: 1.3–1.6× slower

   Double-buffering the registers helped (17.2 → 15.9 µs). Remapping to full-sector loads made BF16 *worse* (→ 17.7 µs).
   **Kept** only for the router at m=8: 5.44 vs 5.79 µs, from the in-register FP32 → hi/lo split.
2. **Triton standard orientation** (`x[BM,BK] @ Wᵀ`, meant to get wgmma) for m=128/256. **No better than swap-AB, and both lose to cuBLAS on the BF16 layers**:
   - qkv_gate m=128: 20.0 vs 16.2 µs
   - pooler m=128: 27.7 vs 24.8 µs
   - qkv_gate m=256: 27.5 vs 18.9 µs

   **Kept** for the router at m=128/256 only (9.2–9.3 µs vs 16.6, 13.7–13.8 µs vs 17.3, as measured by bench.py). The BF16 layers at m > 32 stay on cuBLAS.
3. Router m=256 through swap-AB `gemm` measured 21 µs in bench.py even though tune.py said 14.3. The standard-orientation `mm` config held up in both (13.8 µs). Always confirm a tuned config with bench.py before keeping it.

Open questions / next:
- Profile the CUDA kernel with ncu (needs `DISABLE_DCGM`) to see why it loses: bytes in flight, sector efficiency, LSU stalls.
- m=128/256 BF16 needs a real TMA + wgmma, warp-specialized kernel (CUTLASS/CuTe) to beat cuBLAS. Triton 3.6 didn't get there.
- The large layers sit at ~85–88% of H100 HBM bandwidth (pooler 22.7 µs against a 20.0 µs floor). Triton's gemv at m=1 is the best so far.

## 2026-11-08 · H100 (Rorqual) · muse_bf16_proj
- About **1.15× geomean vs SGLang** on H100, measured with `muse_bf16_proj/bench.py` as a stand-in for the judge (by load: decode1 1.25, verify1 1.39, verify32 0.97). On the closed board the leader had 1.728× on H200.
- m = 128/256 still fall back to cuBLAS, and that's the main gap.
- Lesson: Triton configs tuned with runtime split-K arguments don't carry over to constexpr split-K arguments (the best config differs), so tune and run with the same form.

## Carried over from the kernel track (muse_bf16_proj)
- `../muse_bf16_proj/submission.py`: split-K Triton with swap-AB (W is the MMA's wide operand, tokens the narrow side). FP32 split-K workspace with an atomic-ticket last-CTA reduction, so a graph replay needs no memset. CUDA-core GEMV at m=1. Router as a BF16 hi+lo split at m≥2. Tuned configs live in `tuned_a.json` / `tuned_b.json`.
- From that problem's AGENTS.md: time CUDA-graph replays when tuning. At these sizes eager timing misleads, and configs that win eagerly can lose under graphs. → [[benchmarking-kernels]]
