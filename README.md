# Muse Spark kernel optimization

Custom GPU kernels for the inference hot paths of **Muse Spark Open**, Meta's open MoE model. The problems come from the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev) (`okc`): each kernel has to compute the same result as SGLang's production kernel, within a fixed error bound, and run faster.

This repo has my solutions, the benchmark harness I used to measure them, and the notes I kept while working (`gpu-kernels/`). The work was done after the challenge's kernel track closed, so every result here is measured locally rather than by the official judge.

## Results

Speedup over SGLang's kernel, on **one NVIDIA H100 SXM 80GB**. Higher is better; 1.00× means the same speed.

| Problem | What it computes | Speedup | Status |
|---|---|---:|---|
| [`muse_bf16_proj`](muse_bf16_proj/) | Dense BF16 projections and the FP32 router (GEMV / skinny GEMM) | **1.17×** | done |
| [`muse_attention_decode`](muse_attention_decode/) | Decode and speculative-verify attention over an 80K-token paged FP8 KV cache | **1.44×** | done |
| [`muse_moe_w4a16`](muse_moe_w4a16/) | Routed experts: top-8 of 256, NVFP4 weights, SwiGLU, per-expert norm | **1.60×** | iterating |
| `muse_decode_layer` | A full decode layer | — | not started |
| `muse_prefill_layer` | A full prefill layer | — | not started |

Every number in this README passes the problem's own correctness checks (`reference.py`).

### `muse_bf16_proj`: 1.17×

Speedup per load:

| Load | Tokens per step | Speedup |
|---|---:|---:|
| decode 1 | 1 | 1.25× |
| decode 16 | 16 | 1.15× |
| decode 32 | 32 | 1.16× |
| verify 1 | 8 | 1.39× |
| verify 16 | 128 | 1.08× |
| verify 32 | 256 | 1.04× |
| **Geomean** | | **1.17×** |

**What made it faster**
- **Split-K GEMM in Triton, tuned per shape.** For few tokens, the weight matrix is the operand that fills the tensor core's tall dimension (swap-AB). The last CTA to finish reduces the partial results, using a counter that resets itself, so no separate memset kernel is needed.
- **A CUDA-core GEMV** for single-token calls.
- **The FP32 router** runs as a BF16 high + low pair with FP32 accumulation. The rules allow this.
- **cuBLAS is kept for the BF16 layers at 128 and 256 tokens**, because neither Triton orientation beat it there.

**What didn't work:** a hand-written CUDA kernel that streams weights straight into registers. Nsight Compute showed why: keeping enough bytes in flight in registers limits occupancy, while Triton's shared-memory pipeline reaches full bandwidth with one warp per scheduler.

### `muse_attention_decode`: 1.44×

Per-shape times are in µs (lower is better). Each load is weighted 46 × sliding-window + 16 × full-attention layers per step.

| Load | Layer | SGLang (FA3) µs | Ours µs | Speedup |
|---|---|---:|---:|---:|
| decode 1 | sliding | 22.3 | 8.4 | 2.67× |
| decode 1 | full | 33.3 | 23.7 | 1.41× |
| decode 16 | sliding | 27.8 | 13.2 | 2.11× |
| decode 16 | full | 213.3 | 140.0 | 1.52× |
| decode 32 | sliding | 36.4 | 17.1 | 2.13× |
| decode 32 | full | 543.5 | 273.6 | 1.99× |
| verify 1 | sliding | 24.5 | 13.3 | 1.85× |
| verify 1 | full | 38.3 | 48.5 | 0.79× |
| verify 16 | sliding | 30.5 | 19.4 | 1.57× |
| verify 16 | full | 222.2 | 239.0 | 0.93× |
| verify 32 | sliding | 36.6 | 24.9 | 1.47× |
| verify 32 | full | 424.5 | 466.5 | 0.91× |
| **Geomean over the 6 loads** | | | | **1.44×** |

**What made it faster.** It's a two-kernel design: a vectorized prep kernel (q/k RMSNorm, RoPE, FP8 cache write), then one split-KV attention kernel. The last split for each request and head group merges the partial results and applies the output norm × gate.

The biggest wins, each found with an Nsight Compute profile:
1. **Vectorized split merge.** Merging the partial results one at a time cost 126 dependent L2 round trips.
2. **More CTAs at batch 1.** A 4-CTA grid became 264 or more CTAs.
3. **FP8 query.** Rounding the query to FP8, as FA3 does, removed K's FP8→BF16 conversion.
4. **One CTA reads both KV heads' cache row.** A block-diagonal Q lets it use the whole 128-byte row.

**Known gap:** verify on full-attention layers (8 query tokens per request at 80K context) still runs at 0.79–0.93× of FA3. With 8 query tokens it's close to compute-bound, and needs FA3-class `wgmma` and `exp2` throughput.

### `muse_moe_w4a16`: 1.60×

Speedup over SGLang's Marlin MoE path, per load (one shape per load, times in µs):

| Load | Tokens | SGLang (Marlin) µs | Ours µs | Speedup |
|---|---:|---:|---:|---:|
| decode 1 | 1 | 85.6 | 38.4 | 2.23× |
| decode 16 | 16 | 180.4 | 108.7 | 1.66× |
| decode 32 | 32 | 180.0 | 143.2 | 1.26× |
| verify 1 | 8 | 160.6 | 102.0 | 1.57× |
| verify 16 | 128 | 485.6 | 320.0 | 1.52× |
| verify 32 | 256 | 642.9 | 430.8 | 1.49× |
| **Geomean** | | | | **1.60×** |

**What made it faster.** The design is four raw-CUDA kernels, compiled in-process with NVRTC:
1. **Route:** groups each token's local (token, expert) pairs by expert and writes the work items.
2. **Gate/up GEMM:** computes SiLU(gate)·up in the epilogue.
3. **Down GEMM:** applies the norm weight and accumulates each row's sum of squares in the epilogue.
4. **Norm + weighted sum.**

How the two GEMMs work:
- **Weights are dequantized in registers.** NVFP4 weights go to BF16 on the fly and feed `mma.sync` tensor-core instructions, with the weights as the tall operand and tokens as the narrow one.
- **The weights are repacked once, in `prepare()`.** Each thread's 8 FP4 values for one k-step sit in one 32-bit word. The bits are placed so that a shift and a mask turn them into BF16 values, and one `bf16x2` multiply applies the FP8 block scale. That's about 11 integer ops per 8 weights.
- **A single warp keeps the pipeline full.** It streams each CTA's weight chunk, scales and token rows into a shared-memory ring with `cp.async.bulk` (1-D TMA) and mbarriers.
- **Tuning favours occupancy.** A small ring (2–4 stages) leaves room for more CTAs on each SM, and that matters most because the dequant makes these GEMMs instruction-heavy.

**What didn't help:**
- **Split-K at small batches:** the extra warps just waited on data.
- **8-warp CTAs:** they halve the activation traffic, but 4 warps still won at every batch size.
- **Scheduling the work items of a hot expert together** to share L2: no better.
- **Bigger token blocks:** fewer CTAs fit on each SM, and that cost more than the re-dequantization they save.

**Known gap.** At 3.35 TB/s, a kernel that only streams weights would score about 2.4× here. The remaining time is mostly in two places:
- **Verify 16/32 are instruction-bound.** Hot experts take up to 161 tokens and get re-dequantized once per 16-token item.
- **Decode 1 is limited by data movement.** About 20 µs is the floor; we take 38 µs.

The next step is to dequantize into shared memory once per expert and use `wgmma`, and to prefetch the down-projection weights into L2.

## Methodology

`bench.py` in each problem folder stands in for the official judge, which had closed:

- **CUDA graphs:** each shape is captured as a graph and timed by replaying it. Launch overhead is measured the same way it would be in serving.
- **Cold L2:** a 512 MB buffer is read before every replay, so weights come from HBM, not cache.
- **Paired comparison:** SGLang and the submission alternate in AB/BA pairs on the same fresh inputs. A shape's speedup is the **median of paired ratios**, which cancels drift and ordering effects.
- **Fixed cost removed:** a graph replay's fixed cost (about 4–5 µs) is subtracted from both sides, as the judge does.
- **Scoring:** a load's time is the weighted sum of its shapes. The problem score is the **geometric mean over loads**, so every load counts equally.
- **Correctness:** every scored shape and every judge test is checked against the problem's reference: relative L2 error over the whole output, plus a per-row bound.
- **Baseline:** SGLang at the pinned commit `a55ab7d5`, running its real kernels: FA3 from `sglang-kernel` 0.4.9, Marlin MoE, and cuBLAS. For the MoE baseline, `sgl_stubs.py` stubs the serving-only SGLang modules that its import chain pulls in (vision configs, flashinfer); the Marlin kernels themselves are the real ones.

**Environment:** H100 SXM 80GB (3.35 TB/s HBM3) on the Alliance (Compute Canada) *Rorqual* cluster, PyTorch 2.14.1 + CUDA 13.2, Triton 3.6.

### Caveats
- **Different GPU:** the official leaderboard ran on H200 (4.8 TB/s), and SGLang's baseline gets the same faster memory, so these speedups are not directly comparable to the board.
- **Not the official judge:** results come from my reimplementation of its method, on held-in seeds.
- **Noise:** speedups within about ±2% are inside run-to-run variation.

## Repository layout

```
muse_<problem>/
  TASK.md, reference.py, checks.py, baseline.py   # problem spec and SGLang baseline (from okc init)
  submission.py                                    # the optimized kernel
  bench.py                                         # judge stand-in: tests + timed comparison
  tune.py                                          # config sweeps (graph replays, cold L2)
  prof.py, ncu_summary.py                          # Nsight Compute profiling
  job.sh                                           # Slurm job wrapper for the H100 cluster
gpu-kernels/                                       # knowledge base: concepts, playbooks, findings log
```

## Reproducing

On a machine with an H100 or H200, SGLang at commit `a55ab7d5`, and `sglang-kernel` 0.4.9:

```bash
cd muse_attention_decode
python bench.py                 # tests, then every scored shape against SGLang
python bench.py --only decode1  # one load
```

On a Slurm cluster, use `sbatch job.sh python bench.py`.
