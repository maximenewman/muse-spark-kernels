---
title: Tensor Core Generations (Volta → Ampere → Hopper → Blackwell)
type: concept
tags: [gpu, tensor-cores, architecture, volta, ampere, hopper, blackwell, wgmma, tcgen05]
source_url:
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
  - https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
  - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
---
# Tensor Core Generations

**TL;DR**
- The issue scope grows each generation: **quadpair (8 thr) → warp (32) → warpgroup (128, async) → single thread (async)**.
- Operands leave the register file: RF → RF+SMEM (Hopper B in SMEM) → SMEM+TMEM (Blackwell, D in TMEM).
- Copy engines co-evolve: `cp.async`/LDGSTS (Ampere) → TMA + multicast + clusters/DSMEM (Hopper) → CTA-pair 2-SM MMA (Blackwell).
- Dtypes: FP16 → +INT8/INT4 → +BF16, 2:4 sparse → +FP8 (INT4 dropped) → +MXFP8/6/4, NVFP4.
- Target H200 = Hopper. Write `wgmma` + TMA for sm_90a; tcgen05 doesn't exist there.

## Key ideas / comparison table
| | Volta (sm_70) | Turing (sm_75) | Ampere (sm_80) | Hopper (sm_90a) | Blackwell DC (sm_100a) |
|---|---|---|---|---|---|
| PTX / SASS | `mma` (m8n8k4) / HMMA | `mma` | `mma.sync` (e.g. m16n8k16) / HMMA, `ldmatrix` | `wgmma.mma_async` / HGMMA | `tcgen05.mma` / UTCHMMA, UTCQMMA, UTCOMMA |
| Issuer | quadpair, 8 thr | warp | warp, sync | warpgroup (4 warps), async | 1 thread, async |
| A | RF | RF | RF | RF or SMEM desc | SMEM desc or TMEM |
| B | RF | RF | RF | SMEM desc | SMEM desc |
| C/D | RF | RF | RF | RF | **TMEM** |
| Shape | 8×8×4 per quadpair | — | 16×8×16 | m64 n8–256 k16 | M 64/128 (×2 with 2-SM), N ≤ 256, K = 32 B |
| Dense FP16 FLOPs/clk/SM | 1024 (8 TC × 4×4×4) | — | 2048 (4 TC × 512) | not in sources' text | not in sources' text |
| New dtypes | FP16→FP32 | INT8, INT4 | BF16, TF32, 2:4 sparse | FP8 E4M3/E5M2; INT4 deprecated | MXFP8/6/4, NVFP4; 4:8 pair sparsity for NVFP4 |
| Data movement | ld.shared → RF | — | `cp.async` (LDGSTS), GMEM→SMEM bypassing RF | TMA (`cp.async.bulk.tensor`), clusters, DSMEM, multicast | + CTA pairs (TPC), B split over DSMEM, `tcgen05.cp` SMEM→TMEM |
Sources: SemiAnalysis for all columns. Colfax for Blackwell operands/shapes. Worklog for Hopper shapes. TF32 on Ampere is general knowledge, not from these sources.

- **Why fatter instructions:** issue costs ~30 pJ vs ~1.5 pJ for an HFMA. Bigger MMAs amortize issue/control energy. [semianalysis]
- **Why async + deeper staging:** TC throughput ~doubled each generation while memory latency didn't improve. Each generation needs more bytes in flight, so SMEM grew (RF stayed flat). Blackwell SMEM didn't grow vs Hopper, but 2-SM MMA halves per-SM B traffic. [semianalysis]
- **Why D moved to TMEM:** the accumulator is read and written ~2·K_t times per output tile. Keeping it next to the TC frees the RF for the epilogue. [semianalysis]
- **Quantization effects:** bigger MMA atoms raise arithmetic intensity but worsen tile/wave quantization on small matrices. [semianalysis]
- **Hopper FP8 accumulation:** ~22-bit fixed-point internal accumulation per DeepSeek-V3. Promote periodically. [semianalysis]
- **Blackwell (Colfax):** `.kind::f16 | tf32 | f8f6f4` (+ block-scaled kinds). `idesc` 32-bit instruction descriptor. `enable-input-d` predicate. M=64 needs N%8, M=128 needs N%16. A 128×256 FP32 accumulator = half of TMEM.

## Numbers worth remembering
- Volta 1024 and Ampere 2048 dense FP16 FLOPs/clk/SM. [semianalysis]
- H100 SXM: 132 SMs × 4 TCs, 3.35 TB/s HBM3, 50 MB L2, 256 KB SMEM+L1/SM, 65,536 regs/SM. [worklog]
- Blackwell TMEM: 128 lanes × 512 cols × 4 B = 256 KB/SM. [colfax, semianalysis]
- (Not in these sources: H100/H200 peak TFLOPs, H200 HBM3e bandwidth. Look them up in NVIDIA datasheets before relying on them.)

## Code patterns
Per-generation "one MMA" in PTX:
```
// Ampere+: warp-synchronous, operands in registers (see [[mma-ptx]])
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3};
// Hopper: warpgroup-async, B (and here A) from smem descriptors, D in regs (see [[wgmma]])
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {d0..d63}, descA, descB, 1,1,1,0,0;
wgmma.commit_group.sync.aligned;  wgmma.wait_group.sync.aligned 0;
// Blackwell: single thread, D in TMEM, completion via mbarrier (see [[tcgen05-tmem]])
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], descA, descB, idesc, p_accumulate;
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar];
```

## Pitfalls
- Arch-specific features need the `a` suffix (`sm_90a`, `sm_100a`). Code built for those won't run on other arches. The worklog builds `compute_90a`.
- Consumer Blackwell (sm_120) has no TMEM/tcgen05, so "Blackwell" code isn't portable within the family. [colfax]
- Sparse-FLOP marketing: real 2:4 speedups on Hopper fall well short of 2×. [semianalysis]
- (inference) Porting a Blackwell NVFP4 kernel to Hopper isn't a recompile. There's no FP4 MMA on Hopper, so software dequant to BF16/FP8 is required.

## Relevance to the serving track
- (inference) On H200 the decision is **wgmma vs mma.sync vs FMA** for the inner product, and **TMA vs cp.async vs ld.global** for weights. At m ≤ 32 all three compute paths are far above the HBM roofline. Pick the one with the fewest instructions per weight byte and the cleanest pipeline. wgmma+TMA gets the fewest issuing threads; mma.sync with `ldmatrix` is easier to fuse with dequant (NVFP4 MoE).
- (inference) FP8 on Hopper (wgmma k32) halves weight bytes if a problem tolerates it. That's the main lever generation features offer for memory-bound decode. NVFP4 on Hopper is a storage format only.
- (inference) The "deeper staging each generation" trend is exactly why a decode GEMV must keep many KB in flight per SM. Latency hasn't improved, so bandwidth = bytes_in_flight / latency.

## Related
[[wgmma]] · [[tcgen05-tmem]] · [[mma-ptx]] · [[tma]] · [[thread-block-clusters]] · [[warp-specialization]] · [[semianalysis-tensor-core-evolution]] · [[colfax-blackwell-tmem-gemm]] · [[h100-gemm-worklog-cudaforfun]]
