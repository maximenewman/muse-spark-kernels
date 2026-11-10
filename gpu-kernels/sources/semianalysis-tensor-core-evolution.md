---
title: SemiAnalysis – NVIDIA Tensor Core Evolution, Volta to Blackwell
type: source
tags: [gpu, tensor-cores, volta, ampere, hopper, blackwell, architecture, nvfp4, tmem]
source_url:
  - https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell   # redirect target actually read
---
# SemiAnalysis – NVIDIA Tensor Core Evolution: From Volta to Blackwell (June 23, 2025)

**Read status:** The free portion was read, up to the end of the Blackwell section. The rest is **paywalled**: programming-model evolution (high- to single-occupancy), explicit async execution, warp specialization. The per-generation spec tables are **images** and weren't transcribable, so only numbers stated in body text appear below.

**TL;DR**
- Each generation widened the MMA's issue scope: Volta quadpair (8 threads) → Ampere warp (32) → Hopper warpgroup (128, async) → Blackwell single thread (async, TMEM accumulator, optional 2-SM pair).
- Operands migrated off registers: Volta/Ampere A,B,C in RF. Hopper A in RF or SMEM, B in SMEM, C in RF. Blackwell A,B in SMEM (A may be TMEM per Colfax), D in TMEM.
- The data-movement engines grew alongside: Ampere `cp.async` (LDGSTS, GMEM→SMEM bypassing RF), Hopper TMA (`cp.async.bulk`, single-thread, multicast in clusters), Blackwell CTA pairs sharing B over DSMEM.
- Tensor-core throughput doubled roughly every generation while memory latency didn't improve. Hence deeper SMEM staging, async copies, and larger tiles.
- Data types: FP16 (Volta) → INT8/INT4 (Turing) → BF16 + 2:4 sparsity (Ampere) → FP8 E4M3/E5M2 (Hopper; INT4 deprecated) → MXFP8/6/4 + NVFP4 (Blackwell).

## Key ideas (article order)
1. **First principles.** Amdahl, strong vs weak scaling. Data movement is the "cardinal sin": DRAM cells run at tens of ns vs sub-ns transistor switching. Instruction issue ≈ **30 pJ** vs HFMA ≈ **1.5 pJ** (~20×), which is why fat matrix instructions exist.
2. **Volta (1st gen TC).** PTX `mma` → SASS `HMMA`. 8 TCs/SM in pairs, each 4×4×4 per clock → **1024 dense FLOPs/clk/SM** (FP16 in, FP32 acc). Issue unit is a quadpair (8 threads) computing 8×8×4. Interleaved operand layout. TCs were added late in the design.
3. **Turing (2nd gen).** Adds INT8 and INT4. DLSS.
4. **Ampere (3rd gen).** Warp-wide `mma.sync` (e.g. 16×8×16) and `ldmatrix`. 4 TCs/SM × 512 FLOPs/clk = **2048 dense FLOPs/clk/SM** (FP16). `cp.async` (LDGSTS) copies GMEM→SMEM asynchronously without staging registers. BF16 becomes the de-facto half type (FP32 range, no loss scaling). 2:4 structured sparsity (2× theoretical).
5. **Hopper (4th gen).** `wgmma.mma_async` → SASS `GMMA`/`HGMMA`. Warpgroup = 4 warps = 128 threads, asynchronous. A from RF or SMEM, B from SMEM. Shapes `m64nNk16`, N = 8…256 step 8 (16-bit inputs). FP8 E4M3/E5M2 with FP32 accumulate. Thread-block clusters (CGA) co-schedule CTAs within a GPC and expose **DSMEM**. **TMA**: one thread launches a bulk copy with HW address generation and OOB handling. It has **higher latency for small requests**, so NVIDIA recommends it for large, 16-byte-multiple transfers. Multicast loads one tile into several SMs' SMEM, cutting L2/HBM traffic. FP8 accumulation caveat: DeepSeek-V3 reports a ~22-bit fixed-point accumulation path, so promote to CUDA cores every N_c accumulations.
6. **Blackwell (5th gen).** `tcgen05.mma` → SASS `UTCHMMA`/`UTCQMMA`/`UTCOMMA`. Single-thread issue.
   - **TMEM:** 128 lanes × 512 cols × 4 B = 256 KB, the same size as the RF. Warpgroup access. SW-managed alloc/dealloc/copy. D always lives in TMEM, because D is touched ~2·K_t times per tile (K_t = number of K tiles).
   - **CTA pair:** two CTAs whose cluster ranks differ in the last bit map to one TPC (2 SMs). **MMA.2SM:** the leader CTA issues, M doubles, B is split so each SM loads half of B, and tile exchange goes over DSMEM. Because of this, Blackwell SMEM didn't grow vs Hopper while effective capacity per MMA ~doubles.
   - Weight-stationary mode: a collector buffer caches B for reuse (conv-style).
   - Formats: MXFP8/6/4, **NVFP4**. The article argues NVFP4 is more accurate than MXFP4 due to smaller blocks, its scale format, and two-level scaling. FP8 and FP6 have equal theoretical throughput (likely shared circuits). Pair-wise **4:8 sparsity** for NVFP4, which the article argues is not effectively more relaxed than 2:4.
   - Blackwell Ultra: mentioned with reduced INT8 throughput.
7. **Trends.** TC size grew faster than core count. MMA arithmetic intensity grows linearly with tile size, which favors big tiles. Tile quantization (many small cores) and wave quantization (few big cores) both hurt small matrices. SMEM grew most generations while the RF stayed constant. FP64 is de-emphasized and INT4 deprecated. Real 2:4 sparse GEMM on Hopper falls well short of 2×, and the authors ask NVIDIA to stop quoting sparse FLOPs.

## Numbers worth remembering (only those in visible text)
| Gen | Issue scope | Operands | Dense FLOPs/clk/SM (FP16) | Example shape |
|---|---|---|---|---|
| Volta | 8 threads (quadpair) | RF | 1024 | 8×8×4 per quadpair |
| Ampere | warp (32) | RF (via ldmatrix) | 2048 | m16n8k16 |
| Hopper | warpgroup (128), async | A: RF/SMEM, B: SMEM, C: RF | not stated in text | m64nNk16, N=8..256 |
| Blackwell | 1 thread, async | A,B: SMEM, D: TMEM | not stated in text | 2SM doubles M |
- TMEM 256 KB/SM. Issue energy 30 pJ vs 1.5 pJ HFMA.
- Hopper/Blackwell FLOPs/clk, SMEM sizes, and HBM bandwidths are in images or behind the paywall. **Not captured.**

## Code patterns
The article contains no code. The instruction lineage to grep for in SASS/PTX:
```
Volta/Turing : HMMA.884 / mma.sync.aligned.m8n8k4
Ampere       : HMMA.16816, LDSM (ldmatrix), LDGSTS (cp.async)
Hopper       : HGMMA (wgmma.mma_async), UTMALDG/UTMASTG (TMA cp.async.bulk.tensor), SYNCS (mbarrier)
Blackwell    : UTCHMMA/UTCQMMA/UTCOMMA (tcgen05.mma), LDTM/STTM (tcgen05.ld/st)
```
(SASS names other than HMMA/GMMA/UTC*MMA/LDGSTS are inference from general knowledge, not this article.)

## Pitfalls
- TMA has high latency on small requests (stated). Don't use it for tiny scattered loads.
- Sparse FLOPs are marketing. Real 2:4 sparse kernels on Hopper fall well short of 2× (stated).
- FP8 wgmma accumulation has reduced precision (~22-bit fixed point per DeepSeek-V3). Long-K FP8 GEMMs need periodic promotion to FP32 on CUDA cores.

## Relevance to the serving track
- (inference) On H200 the relevant generation is Hopper: `wgmma` (BF16/FP8), TMA, clusters/DSMEM. There's no FP4 tensor path, so NVFP4 MoE weights on H200 must be dequantized (FP4→BF16/FP8 in registers or SMEM) before `wgmma`. Since the kernels are memory-bound, the win is moving 4-bit bytes from HBM; dequant ALU cost is mostly hidden.
- (inference) "TMA has higher latency for small requests" matters directly. For m=1–16 GEMV-like shapes, the x vector load is tiny. Weight tiles should be loaded in large TMA boxes (≥ several KB) with many stages in flight to cover HBM latency. A plain `ld.global.v4`/`cp.async` path can be competitive for the tiny operand.
- (inference) Router FP32 via BF16 hi+lo split (3 BF16 products, FP32 acc) triples MMA work. Irrelevant when memory-bound, but it doubles weight bytes if W is stored as hi+lo. Better to load FP32 W once and split in registers.
- (inference) Wave quantization (stated as a general problem for big cores on small matrices) is the main issue for small-n layers like router/pooler. Use split-K or stream-K so all 132 SMs stream weights.

## Related
[[tensor-core-generations]] · [[wgmma]] · [[tcgen05-tmem]] · [[thread-block-clusters]] · [[tma]] · [[mma-ptx]] · [[warp-specialization]] · [[colfax-blackwell-tmem-gemm]] · [[h100-gemm-worklog-cudaforfun]]
