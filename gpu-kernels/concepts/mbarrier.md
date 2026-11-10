---
title: mbarrier (shared-memory async barrier)
type: concept
tags: [gpu, hopper, sm90, mbarrier, tma, synchronization, memory-model, async-proxy]
source_url:
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html   # §9.7.15.16 mbarrier, fence.mbarrier_init
  - https://yang-yifan.github.io/blogs/memory_model/memory_model.html
  - https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
---
# mbarrier (shared-memory async barrier)

**TL;DR**
- 8-byte (`.b64`, 8 B-aligned) opaque object in shared memory; tracks **phase**, **pending arrivals**, **expected arrivals**, and (sm_90+) **tx-count** (outstanding async bytes).
- A phase completes when pending arrivals == 0 **and** tx-count == 0; then it auto-reinitializes for the next phase (phase parity flips).
- TMA loads use it: `arrive.expect_tx(bytes)` raises tx-count, TMA's complete-tx lowers it; consumers `try_wait.parity` on the phase bit.
- Default semantics: `arrive` = **release**, `try_wait` = **acquire** (scope `.cta` default; `.cluster` available). TMA completion = release at cluster scope + implicit async-proxy fence ⇒ no extra fence needed to read TMA-loaded smem.
- Unlimited count (only smem-limited), unlike the 16 hardware named barriers; enables split arrive/wait so work can happen in between.

## Key ideas
- **Contents** (PTX): current phase; pending arrival count; expected arrival count; tx-count. `.layout::v1` (newer) adds a payload report and a "conditional phase". Ranges (`.layout::v0`): expected/pending arrivals up to 2^20−1, tx-count in [−(2^20−1), 2^20−1]. (v1: arrivals up to 2^9−1.)
- **Lifecycle**: `mbarrier.init` before use; any other op on uninitialized object = UB; non-mbarrier access to an initialized object = UB; `mbarrier.inval` before repurposing memory.
- **Phase rule**: for each phase, at least one `test_wait/try_wait` must return True before an arrive in the next phase. Waits are valid only for the current incomplete phase (returns False) or the immediately preceding phase (returns True) ⇒ parity (1 bit) suffices, but you must track phase through the lifetime.
- **expect-tx / complete-tx**: expect-tx(n) increases tx-count by n; complete-tx(n) (performed by `cp.async.bulk*` with `.mbarrier::complete_tx::bytes`) decrements and may complete the phase. tx-count may go transiently negative (range allows it), so TMA bytes can arrive before expect_tx is posted (inference from the signed range — ordering expect before issue is still the standard pattern).
- **`mbarrier.arrive.expect_tx`**: expect-tx then arrive with count 1 in one instruction (CuTe `set_barrier_transaction_bytes`, libcu++ `barrier_arrive_tx`). Standalone `mbarrier.expect_tx` (relaxed only) also exists.
- **try_wait vs test_wait**: test_wait non-blocking; try_wait (sm_90+) may suspend the thread until phase completes or a system-dependent time limit (optional `timeHint` ns) — still needs a loop.
- **Cluster**: threads can *arrive* on a remote CTA's mbarrier (`.shared::cluster`, no returned state) but can only *wait* on local ones. TMA multicast signals the barrier at the same smem offset in each destination CTA. After init, other CTAs must observe it: `fence.mbarrier_init.release.cluster` + cluster barrier (CuTe: `cluster_sync(); fence_barrier_init();`).
- **Acquire guarantees** (PTX): when an acquire try_wait returns True, the waiter sees (a) all non-async accesses before release-arrives in that phase, (b) cp.async ops before `cp.async.mbarrier.arrive`, (c) all `cp.async.bulk` ops using that mbarrier.
- **Memory model patterns** (Yang memory_model):
  - Execution order ≠ memory order: `arrive.relaxed`/`try_wait.relaxed` only order execution; add `fence.release/acquire.cta` or use default release/acquire.
  - Generic→generic intra-CTA: `st.shared; mbarrier.arrive.release.cta` / `mbarrier.try_wait.acquire.cta; ld.shared` (lowers to MEMBAR.CTA-equivalent).
  - Intra-cluster via DSMEM: `st.async` (implicit release, cluster scope, DSMEM-only, arrives on remote mbarrier) + `try_wait.acquire` avoids MEMBAR.GPU that `fence.cluster`/`barrier.cluster` emit.
  - **Generic→async (CUDA core writes smem, TMA store/wgmma reads it)**: `st.shared; fence.proxy.async.shared::cta; mbarrier.arrive` → consumer `try_wait` then issues TMA/MMA. Lowers to `FENCE.VIEW.ASYNC.S`.
  - **Async→generic (TMA load → threads)**: nothing extra; TMA completion arrives with release and fences the async proxy.
  - Async→async (TMA → tensor core): same as above, consumer thread waits then issues MMA.
- Synchronization table (Yang): `bar.sync` = exec+mem, CTA, generic; mbarrier = exec + optional mem, CTA/cluster, generic/async; `fence.proxy.async` = mem only, generic→async.

## Numbers worth remembering
- Size/alignment 8 B / 8 B. Arrival/tx ranges 2^20−1 (`.layout::v0`).
- PTX versions: `try_wait` PTX 7.8 / sm_90; `.relaxed` PTX 8.6; `.cluster` scope sm_90.

## Code patterns
```ptx
// init (one thread), make visible, then use
mbarrier.init.shared::cta.b64 [bar], 1;               // expected arrivals per phase
fence.mbarrier_init.release.cluster;                   // needed if other CTAs / TMA multicast will signal it
// (then __syncthreads() or barrier.cluster.arrive/wait)

// producer (elected thread) per stage
mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], BYTES;
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dst], [tmap, {c0, c1}], [bar];

// consumer: spin on parity
WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [bar], phase;
@!p bra WAIT;
// phase ^= 1 when the ring index wraps (track per stage)

// consumer -> producer "slot free" (empty barrier, count = #consumer warps/threads)
mbarrier.arrive.shared::cta.b64 _, [empty_bar];
```
```cpp
// CuTe (from Yang's TMA post)
initialize_barrier(mbar, 1); set_barrier_transaction_bytes(mbar, bytes);   // thread 0
__syncthreads(); wait_barrier(mbar, /*phase*/0);
```

## Pitfalls
- Wrong parity / not flipping phase on ring wrap ⇒ deadlock or reading stale stage.
- expect_tx bytes ≠ actual bytes delivered (e.g. two TMA boxes per stage but only one counted) ⇒ hang or premature completion.
- Reusing a stage's smem for the next TMA before all consumers finished reading ⇒ need an "empty" mbarrier (consumer arrive → producer wait) — the full/empty pair pattern of [[warp-specialization]].
- Thread-written smem consumed by TMA store/wgmma without `fence.proxy.async` ⇒ stale data.
- Remote barrier init not fenced before multicast ⇒ UB.
- Re-initializing an in-use mbarrier without `mbarrier.inval` ⇒ UB.

## Relevance to the serving track
- For a skinny weight-streaming GEMM, the pipeline is just N stages of {full_bar, empty_bar} around W boxes; at m ≤ 32 consumers finish fast, so the ring depth (bytes in flight) — not compute — sets throughput. (inference)
- mbarrier init is part of the prologue: put it before `griddepcontrol.wait` under [[pdl]]. (inference)
- For a tiny CTA (1–4 warps) with a single-shot load of all of its W slice, one mbarrier with one expect_tx for all boxes is the minimal-latency form (Yang's single-load example generalised). (inference)

## Related
[[tma]] [[pdl]] [[warp-specialization]] [[wgmma]] [[yangyifan-cute-tma]]
