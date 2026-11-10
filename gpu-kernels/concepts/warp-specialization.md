---
title: Warp Specialization (producer/consumer warpgroups, mbarrier pipelines, setmaxnreg)
type: concept
tags: [gpu, hopper, blackwell, warp-specialization, mbarrier, setmaxnreg, pipelining, tma]
source_url:
  - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
  - https://github.com/pranjalssh/fast.cu  (h100/matmul/matmul_8.cuh)
  - https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
---
# Warp Specialization

**TL;DR**
- Split a CTA into roles: a **producer** warpgroup (one elected thread issues [[tma]] loads) and one or more **consumer** warpgroups (issue [[wgmma]] and run the epilogue). They communicate through a ring of SMEM stages guarded by `full[]`/`empty[]` **mbarriers**.
- TMA completion is tracked by mbarrier **transaction bytes** (`arrive.expect_tx` + `complete_tx::bytes`), so no thread polls data.
- `setmaxnreg` moves registers from the producer (24–40 regs) to the consumers (232–240). Worklog: 610 → 631 TFLOPs.
- Worklog gain from specialization alone: 423 → 498 TFLOPs (loads ~1415 cycles overlapped with TC ~703 cycles).
- Blackwell takes it further: a single MMA-issuing thread plus TMEM-reading epilogue warps ([[tcgen05-tmem]]).

## Key ideas
- **Why.** On Hopper the load engine (TMA) and the MMA unit are both async and issued by very few threads. Dedicating warps to each role lets loads for stage i+1..i+Q run while the tensor cores chew stage i. [worklog]
- **Ring protocol.** Producer: `wait(empty[q])` → `expect_tx(full[q], bytes)` → TMA(…, `full[q]`). Consumer: `wait(full[q])` → MMA → `arrive(empty[q])`. Barriers start so that producers see every slot empty (consumers pre-arrive on all `empty[q]` at start in fast.cu). [worklog]
- **Phase/parity.** An mbarrier completes phases 0,1,0,1… `try_wait.parity` waits for the phase with the given parity. Each side keeps `(qidx, p)` and flips `p` when `qidx` wraps. [fast.cu]
- **Token-less PTX barriers.** Replacing `cuda::barrier` (arrival tokens) with raw `mbarrier.try_wait.parity` cut the sync count ~257 → 3 and gave 660 → 704 TFLOPs. [worklog]
- **Multiple consumers.** Two consumer warpgroups each own half of BM (e.g. 2×64 rows of a 128×256 tile). This halves per-thread accumulator registers and avoids wgmma serialization from spills (123 → 610). [worklog]
- **setmaxnreg.** `setmaxnreg.dec.sync.aligned.u32 N` (producer) and `.inc` (consumers) rebalance the per-warp register allocation at runtime. The budget must fit 65,536 regs/SM: 240·128·2 + 24·128 = 64,512. [worklog]
- **Blackwell variant.** One elected thread issues all `tcgen05.mma` for a k-block, `tcgen05.commit` arrives on an mbarrier when the MMAs finish, and epilogue warps wait on it before `tcgen05.ld`. [colfax]
- The SemiAnalysis sections on warp specialization and the move from high- to single-occupancy are **paywalled** (not read).

## Numbers worth remembering
- H100: 65,536 regs/SM, max 255/thread, ≤ 1024 threads/SM. [worklog]
- Worklog config: 3 warpgroups (384 threads) = 1 producer + 2 consumers. QSIZE 5 at 128×128×64, QSIZE 3 at 128×256×64 (SMEM limited). Regs: producer 24 (32 if >2 consumers), consumers 240 (2 consumers) / 160 (3) / 256 (1). [fast.cu]
- Cycles per block before overlap: load 1415, TC 703, store 4572. [worklog]

## Code patterns (fast.cu)
```cuda
// mbarrier init (one thread), then make it visible cluster/CTA-wide
asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar_ptr), "r"(count));
// full[i]: count 1 (producer's expect_tx arrival); empty[i]: count = num_consumers * cluster_size

// producer: declare bytes the TMA will deliver, and arrive (1 arrival)
asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar_ptr), "r"(bytes));
// TMA load signals the same barrier with complete_tx when bytes land
"cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3,%4,%5}], [%2];"

// wait: spin on parity
asm volatile("{\n .reg .pred P1;\n LAB_WAIT:\n"
  "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
  "@P1 bra.uni DONE;\n bra.uni LAB_WAIT;\n DONE:\n }\n" :: "r"(mbar_ptr), "r"(phase));

// consumer release
asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n" :: "r"(mbar_ptr), "r"(1) : "memory");

// register rebalancing (whole warpgroup executes it)
asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(24));   // producer
asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(240));  // consumers
```
Role dispatch:
```cuda
int wg_idx = threadIdx.x / 128, tid = threadIdx.x % 128;
if (wg_idx == 0) { setmaxnreg_dec<24>(); if (tid == 0) producer_loop(); }
else             { setmaxnreg_inc<240>(); consumer_loop(wg_idx - 1); }
```
Consumer-only sync for the epilogue uses a named barrier so the producer isn't involved: `bar.sync 10, 256;` [fast.cu kernel 10]

## Pitfalls
- Phase bookkeeping bugs deadlock or silently read stale stages. Producer and consumer must flip parity at the same wrap point.
- `expect_tx` byte count must exactly equal the bytes all TMAs targeting that barrier deliver (including multicast copies landing in this CTA). Otherwise the barrier never completes or completes early (inference from tx-count semantics).
- `setmaxnreg` values must be multiples of 8 in [24, 256], and the whole warpgroup must execute it (PTX ISA; not stated in sources). The total must fit the SM register file at the launched occupancy.
- `__syncthreads()` after the role split includes the producer. Use named barriers (`bar.sync id, nthreads`) for consumer-only sync.
- Blackwell: don't overwrite SMEM operands or read TMEM before the `tcgen05.commit` → mbarrier completes. [colfax]

## Relevance to the serving track
- (inference) For memory-bound skinny GEMM the consumer is nearly idle. The design question becomes **how many bytes are in flight per SM**, i.e. stage count × stage size, not TC overlap. A producer thread issuing large TMA boxes into a 4–8+ deep ring is the right skeleton. With m ≤ 32 one consumer warpgroup is plenty.
- (inference) setmaxnreg is irrelevant at small m (accumulators are 4–16 regs/thread). Better to spend occupancy on **2+ CTAs/SM** or multiple producer threads/TMA streams so HBM requests stay saturated.
- (inference) Prologue cost dominates a 2–5 µs kernel: mbarrier init + `fence.mbarrier_init`/cluster sync + `prefetch.tensormap` + first TMA latency. With [[pdl]], do barrier init and descriptor prefetch before `griddepcontrol.wait` so it overlaps the previous kernel's tail.
- (inference) Split-K across warpgroups or CTAs (reduce in SMEM/DSMEM or with atomics) is the natural extension when n/64 tiles < SM count (router, pooler). See [[persistent-kernels-and-tile-scheduling]].

## Related
[[wgmma]] · [[tma]] · [[tcgen05-tmem]] · [[thread-block-clusters]] · [[persistent-kernels-and-tile-scheduling]] · [[pdl]] · [[register-tiling]] · [[h100-gemm-worklog-cudaforfun]] · [[colfax-blackwell-tmem-gemm]]
