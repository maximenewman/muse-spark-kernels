# Muse Spark kernel optimization

Custom GPU kernels for the inference hot paths of **Muse Spark Open**, Meta's open MoE model. The problems come from the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev) (`okc`): each kernel has to compute the same result as SGLang's production kernel, within a fixed error bound, and run faster.

This repo has my solutions, the benchmark harness I used to measure them, and the notes I kept while working (`gpu-kernels/`). The work was done after the challenge's kernel track closed, so every result here is measured locally rather than by the official judge.

## Results

Speedup over SGLang's kernel, on **one NVIDIA H100 SXM 80GB**. Higher is better; 1.00× means the same speed.

| Problem | What it computes | Speedup | Status |
|---|---|---:|---|
| [`muse_bf16_proj`](muse_bf16_proj/) | Dense BF16 projections and the FP32 router (GEMV / skinny GEMM) | **1.17×** | done |
| [`muse_attention_decode`](muse_attention_decode/) | Decode and speculative-verify attention over an 80K-token paged FP8 KV cache | **1.44×** | done |
| [`muse_moe_w4a16`](muse_moe_w4a16/) | Routed experts: top-8 of 256, NVFP4 weights, SwiGLU, per-expert norm | — | in progress |
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

## Methodology

`bench.py` in each problem folder stands in for the official judge, which had closed:

- **CUDA graphs:** each shape is captured as a graph and timed by replaying it. Launch overhead is measured the same way it would be in serving.
- **Cold L2:** a 512 MB buffer is read before every replay, so weights come from HBM, not cache.
- **Paired comparison:** SGLang and the submission alternate in AB/BA pairs on the same fresh inputs. A shape's speedup is the **median of paired ratios**, which cancels drift and ordering effects.
- **Fixed cost removed:** a graph replay's fixed cost (about 4–5 µs) is subtracted from both sides, as the judge does.
- **Scoring:** a load's time is the weighted sum of its shapes. The problem score is the **geometric mean over loads**, so every load counts equally.
- **Correctness:** every scored shape and every judge test is checked against the problem's reference: relative L2 error over the whole output, plus a per-row bound.
- **Baseline:** SGLang at the pinned commit `a55ab7d5`, running its real kernels: FA3 from `sglang-kernel` 0.4.9, Marlin MoE, and cuBLAS.

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
