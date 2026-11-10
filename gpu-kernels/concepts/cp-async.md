---
title: "cp.async (Ampere async global→shared copy), commit/wait groups, multistage pipelining"
type: concept
tags: [gpu, cp-async, pipelining, multistage, ampere, hopper, memory]
source_url:
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html   # §9.7.10.28.3 Non-bulk copy: cp.async, commit_group, wait_group
  - https://github.com/Bruce-Lee-LY/cuda_hgemm   # src/common/ptx.h, src/mma/mma_async_stage3.cu
  - https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
---
# cp.async and multistage pipelining

**TL;DR**
- `cp.async.{ca,cg}.shared.global [dst], [src], cp-size{, src-size}` (sm_80+) starts a **non-blocking** 4/8/16-byte global→SMEM copy that **bypasses registers**. `.cg` (L2 only, 16 B only) is the GEMM default.
- **Completion is per thread, via groups:**
  - `cp.async.commit_group` closes the thread's pending copies into a group.
  - `cp.async.wait_group N` blocks until at most the N most recent groups are pending.
  - **After the wait you still need `__syncthreads()`**, so other threads' copies become visible to your `ldmatrix`.
- **Multistage ring:** with `S` SMEM stages, prefetch `S-1` tiles, then each iteration does wait(`S-2`) → `__syncthreads` → compute stage `i` → issue tile `i+S-1` → commit.
- `src-size < cp-size` zero-fills the rest. That's the free way to predicate K/M tails.
- Turing has none of this. alexarmbr fakes it with an LDG-to-registers prefetch. Hopper adds [[tma]] (`cp.async.bulk`, mbarrier completion) for large tiles.

## Key ideas (PTX ISA §9.7.10.28.3, unless marked)
- **Syntax:**
  - `cp.async.ca.shared{::cta}.global{.L2::cache_hint}{.L2::64B|128B|256B} [dst], [src], cp-size{, src-size | , ignore-src}{, cache_policy}` with cp-size ∈ {4, 8, 16}.
  - `.cg` only allows cp-size 16.
  - `.ca` caches at all levels including L1; `.cg` caches only in L2. Both are hints.
- **`src-size`** (≤ cp-size): copies `src-size` bytes and zero-fills the rest of `dst`. **`ignore-src`** (predicate, PTX 7.5+) copies zeros entirely. Either one bounds-checks a tail tile without branches around the copy.
- **`.L2::128B` prefetch_size**: a hint to also pull 64/128/256 B into L2 (needs PTX 7.4+). The Bruce-Lee macros guard this with `#if CUDA ≥ 11.4`.
- **`.L2::cache_hint` + `cache_policy`** (from `createpolicy.fractional.L2::evict_last/evict_first...`): a per-access eviction hint. Inference: mark streamed-once weights `evict_first` so they don't evict activations or KV that are reused.
- **Ordering:**
  - There's "no ordering guarantee between two cp.async operations" unless they're synchronized with wait_group, wait_all or mbarrier.
  - Two copies in the same group writing the same location is UB.
  - `wait_group` gives no ordering for *other* memory ops.
- **Visibility:** writes are visible to the *executing thread* after `wait_group`/`wait_all` or an mbarrier test/try_wait. Making them visible to the CTA needs a barrier (`__syncthreads`). That's why every pipeline in [[bruce-lee-mma-ptx]] does `CP_ASYNC_WAIT_GROUP(n); __syncthreads();`.
- **Empty groups** are trivially complete. Committing an empty group each iteration in the tail keeps the wait count uniform. That's the CUTLASS-style idiom (inference: not shown in these sources, but follows from the semantics).
- `cp.async.wait_all` ≡ `commit_group; wait_group 0`.
- cp.async can also arrive on an **mbarrier** (`cp.async.mbarrier.arrive`) instead of groups. That's useful for warp-specialized producer/consumer pipelines (from the PTX ISA's mbarrier examples; not explored in these sources).

### Pipeline depth
- **Double buffer** (`pg2s` in Bruce-Lee): load tile k+1 while computing tile k, then `wait_group 0` + sync each iteration.
- **Multistage** (`stage2/3/4`):
  - Keep `K_STAGE-1` groups in flight with `wait_group(K_STAGE-2)`. With `K_STAGE=3`: `CP_ASYNC_WAIT_GROUP(1)`.
  - Ring slots are indexed `(idx+1) % K_STAGE` (`mma_async_stage3.cu` L149–150, L267–268, L308–309).
  - Combine with a **register double buffer** (`ps2r`): `ldmatrix` k-step j+1 while the mma runs k-step j.
- **Register-staged prefetch** (pre-Ampere, [[hidden-treasures-3-gemm-code]] kernel 4):
  - `tileMemcpyLoad` (LDG into `float4` regs), then compute, `__syncthreads`, then `tileMemcpySwizzleStore`.
  - Costs registers (104 → 166 regs/thread in alexarmbr) and two barriers per tile. cp.async removes the register cost.
- SMEM budget = `stages × (BM + BN) × BK × bytes`. E.g. Bruce-Lee stage3: (256+128)·32·2·3 = 73,728 B.

## Numbers worth remembering
- cp-size: 4/8/16 B; `.cg` is 16 B only. sm_80+. PTX ISA 7.0+ (hint qualifiers 7.4, ignore-src 7.5, `::cta` 7.8).
- Bruce-Lee stage3 uses `K_STAGE 3`, `CHUNK_K 2` (32 halves of K per stage), 16 B per thread per copy, 8 warps.

## Code patterns
```cuda
// Bruce-Lee-LY src/common/ptx.h (verbatim)
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
```
Predicated tail (inference; from the PTX `src-size` operand):
```cuda
uint32_t n = in_bounds ? 16 : 0;   // bytes actually read; rest of the 16 B dst is zero-filled
asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_addr), "l"(gptr), "r"(n));
```
Generic S-stage loop (inference; the shape of `mma_async_stage3.cu` L147–342):
```cuda
for (int s = 0; s < S - 1; ++s) { issue_tile(s, slot(s)); cp_async_commit(); }   // prologue
for (int k = 0; k < KT; ++k) {
  cp_async_wait<S - 2>();      // tile k has landed (for this thread)
  __syncthreads();             // ... and for every thread; also frees slot (k-1)%S for reuse
  issue_tile(k + S - 1, slot(k + S - 1));  // may be past the end → empty/zero-filled
  cp_async_commit();           // always commit (possibly empty) to keep group arithmetic uniform
  compute_from(slot(k));       // ldmatrix + mma
}
```

## Pitfalls
- Forgetting `__syncthreads()` after `wait_group`. Each thread waits only on **its own** copies.
- Reusing a slot before every warp finished reading it. The barrier at the top of the next iteration must come before you issue into a slot that was read in the last iteration.
- Addresses: `dst` is a shared-window address (`__cvta_generic_to_shared`), and `src` is global. 16 B copies need 16 B-aligned src and dst.
- `wait_group` takes an immediate (`"n"`). The stage count must be compile-time.
- Inside a CUDA graph nothing changes, but a cold L2 (the challenge timing) makes the **prologue latency** a visible fraction of a 2–17 µs kernel. Deeper stages per SM, or more CTAs, hide it (inference).

## Relevance to the serving track
- **Transfers (inference):** for a weight-streaming skinny GEMM on H100/H200, the goal is **enough bytes in flight per SM** to cover HBM latency. Rough Little's-law: BW × latency, which at multi-TB/s and ~µs-scale latency means tens of KB in flight per SM. Ways to get there:
  - multistage cp.async rings (3–6 stages of W tiles, e.g. BN=64–128 rows × BK=64–128 bf16)
  - several CTAs per SM
  - or [[tma]] bulk copies
- The x/activation tile is tiny (m ≤ 256 rows × BK) and reused across all of the CTA's W rows. Load it once per K stage (or keep all of x[m, Kslice] in SMEM when split-K makes the K slice small).
- **Alternative with no SMEM for W:** direct LDG.128 into registers in mma-fragment order (see the k-permutation trick in [[mma-ptx]]). It skips cp.async/ldmatrix entirely, and loads in flight come from unrolling. Whether it beats cp.async on H200 needs measuring → [[findings-log]].
- **Does not transfer:** the deep-K-loop amortization assumptions of big GEMM. With split-K, each CTA may see only 2–8 K tiles, so prologue/epilogue dominate. Pipeline depth must be traded against K-tiles per CTA (inference).

## Related
[[mma-ptx]] · [[bruce-lee-mma-ptx]] · [[hidden-treasures-3-gemm-code]] · [[alexarmbr-tensor-core-hgemm]] · [[tma]] · [[mbarrier]] · [[warp-specialization]] · [[gemm-tiling-hierarchy]] · [[gemv-and-skinny-gemm]] · [[split-k-and-stream-k]] · [[memory-bandwidth-and-latency]]
