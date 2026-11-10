---
title: "Hidden-Treasures 3-GEMM: alexarmbr matmul-playground code walkthrough (kernel1–6)"
type: source
tags: [gpu, gemm, hgemm, mma-ptx, ldmatrix, swizzling, double-buffering, code-walkthrough, turing]
source_url:
  - raw/hidden-treasures-llms/3-GEMM/   # local copy, read-only, not executed
  - https://github.com/alexarmbr/matmul-playground
---
# Hidden-Treasures 3-GEMM code (alexarmbr matmul-playground)

**TL;DR**
- Local copy of the code behind [[alexarmbr-tensor-core-hgemm]]: six FP16 HGEMM kernels (`src/kernel1.cu`…`kernel6.cu`) plus a cuBLAS baseline. Built for **sm_75 (T4)** with `-Xptxas -dlcm=cg` (L1 off for global loads). The README claims kernel 6 is about 95% of `cublasHgemm` at 8192³.
- Each kernel differs from the previous one by a small, local diff:
  1. Scalar copy.
  2. `float4` unrolled copy.
  3. XOR-swizzled SMEM plus swizzled `ldmatrix` addresses.
  4. Register prefetch of the next K tile.
  5. Bigger tiles (256×256×32), a hand-unrolled `ldmatrix.x4` with XOR stepping, and `#pragma unroll`.
  6. SMEM double buffer, which drops one `__syncthreads`.
- All kernels use `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` (FP16 accumulate). B is row-major `[K,N]`, so it's loaded with `ldmatrix ... .trans`.
- No bounds checks: dimensions must be multiples of the tiles (`assert(M % BM_dim == 0)`). The epilogue reads C and writes D straight from fragments, uncoalesced.

## Key ideas: file map
| File | What it holds |
|---|---|
| `src/structs_n_stuff.cuh` | `CUDA_CHECK`, `sgemm_params {A,B,C,D,alpha,beta,M,N,K}` (L25–36), and `KernelLogger`. The logger uses cudaEvent timing per launch with a **100 ms sleep after every run** (L74–82), and GFLOP/s = 2MNK/t (L83–92) |
| `src/main.cu` | `runner <kernel_id> <iters> M N K`. Ids 1–6, or 99 for cuBLAS (L31–53). CPU verification is off by default (`check_on_cpu = false`, L13) |
| `src/host_utils.cuh` | Allocates A,B,C,D as half, random in [0,1] with `srand(1234)` |
| `src/cublas_benchmark.cu` | `cublasHgemm` baseline |
| `src/device_utils.cuh` | All copy helpers and PTX wrappers (below) |
| `profile.sh` | Locks clocks (`--lock-gpu-clocks=1590`, `--lock-memory-clocks=5000`). Runs each kernel 50× at 1024, 2048, 4096, 8192 and 16384 |
| `CMakeLists.txt` | `CMAKE_CUDA_ARCHITECTURES 75`, `-Xptxas=-v` (prints regs), `-dlcm=cg`, `--generate-line-info`, `--keep` |
| `scripts/shmem_layout_viz.py` | Prints which banks hold which `A[row, cols]` packed element. Counts actual vs ideal wavefronts for a column access, and applies `SWIZZLE_A/B/C` functions to visualize the permutation |

### device_utils.cuh: helper progression
| Helper (lines) | Behavior |
|---|---|
| `tileMemcpy` (L11–35) | Scalar half copy. Threads flattened. Each thread takes a column; rows stride by `num_threads/tile_cols`. Coalesced, but 2 B per request (`LDG.U16`) |
| `tileMemcpyUnrolled` (L38–67) | Same, with compile-time sizes and `#pragma unroll` |
| `tileMemcpyUnrolledVectorized` (L70–105) | Reinterprets as `float4`: 8 halves = 16 B per thread per iteration (`LDG.128`/`STS.128`) |
| `tileMemcpySwizzle<ROWS,COLS,THREADS,SWIZZLE_BITS>` (L109–150) | Vectorized copy with destination float4 index `dst ^= (dst & (0b111<<SWIZZLE_BITS)) >> SWIZZLE_BITS`. The 3 row bits are XORed into the 16-byte-chunk column bits |
| `tileMemcpySwizzleA` (L154–198) | Special case for a 32-half-wide A tile (4 float4 per row): two XOR stages, `(i&0b10000)>>4` then `(i&0b1100)>>2` |
| `tileMemcpyLoad<...,ELEMENTS_PER_THREAD>` (L203–241) | **Global→register only** (`float4 dst_reg[N]`). Comment: "does not result in memory latency stalls", because the stall moves to first use |
| `tileMemcpySwizzleStore` / `tileMemcpySwizzleStoreA` (L294–335 / L379–423) | Register→SMEM half of the split copy, with the swizzle |
| `cvta_to_shared_u32` (L429–438) | Generic→32-bit shared address via `cvta.to.shared.u64` + `cvt.u32.u64` |
| `ldmatrix_m16n8` / `ldmatrix_n8k8` (L443–483) | `ldmatrix .x2` (A) and `.x1.trans` (B) wrappers |
| `mma_sync_m16n8k8` (L485–508) | Correctly comma-separated `mma` wrapper (A: 2 regs, B: 1 reg, C/D: 2 regs of f16x2) |
| `stmatrix_m16n8` / `ldmatrix_m16n8_gmem` (L521–559) | Epilogue. Lane → (row = lane/4, col = lane%4) 4-byte accesses, rows r and r+8. Comment: "this is innefficient, access pattern results in bad coalescing". The comment also notes `stmatrix` needs sm_90 |
| `stmatrix_m16n8_swizzle` (L563–598) | Swizzled SMEM-staged epilogue (`^ ((off & 0b11100000) >> 3)`). Not used by kernels 1–6 |

## Per-kernel diff walkthrough
**Kernel 1: hierarchical tiling** (`kernel1.cu`)
- Launch (L195–255):
  - Tiles: BM=256, BN=128, BK=64. Warps 4 (M) × 2 (N) = 8 warps = 256 threads, laid out as `blockDim(64,4)`.
  - Warp tile: WM=64, WN=64, WK=32. `WARPS_PER_BLOCK_K=2` is only used to derive WK; there's no intra-block split-K.
  - SMEM: (256·64 + 64·128)·2 B = 48 KB. `cudaFuncSetAttribute(...MaxDynamicSharedMemorySize, 65536)`.
- Registers (L55–61): `acc[4][8][2]` u32 (64 regs of packed f16 accumulators), `A_register[4][4][2]`, `B_register[4][8]`.
- Main loop (L75–155):
  - `tileMemcpy` A and B (L79–80), `__syncthreads`.
  - For each `warp_k`: `ldmatrix.x2` all A mma tiles (L93–113). The per-lane address is row `lane%16` of the 16×8 tile.
  - `ldmatrix.x1.trans` all B tiles (L116–130). The address is row `lane%8` of the 8×8 tile in a `[K][N]` buffer.
  - Triple loop k→n→m of `mma.sync.m16n8k8` (L133–152).
  - `__syncthreads` at L154.
- Epilogue (L160–192): `ldmatrix_m16n8_gmem` reads C, computes `acc*α + C*β` in half, then `stmatrix_m16n8` writes D.

**Kernel 2: vectorized + unrolled gmem→smem.** `diff kernel1 kernel2` touches only L80–81:
- `tileMemcpy(...)` becomes `tileMemcpyUnrolledVectorized<BM_dim,BK_dim,NUM_THREADS>` and `<BK_dim,BN_dim,NUM_THREADS>`.
- Buys: 16-byte loads and compile-time-unrolled loops, so several LDGs are in flight. The blog reports ~3×.

**Kernel 3: SMEM swizzle.** The diff vs kernel 2:
- L36–41 add `SWIZZLE_BITS_A = log2(BK/8)`, `SWIZZLE_BITS_B = log2(BN/8)`, and masks `0b1110000 << bits` (byte-offset domain: 3 row bits).
- L85–86 switch the copies to `tileMemcpySwizzle<..., SWIZZLE_BITS_*>`.
- L109–113 and L130–131 apply the same permutation to each lane's `ldmatrix` byte offset: `off ^= (off & MASK) >> BITS`.
- Buys: removes the 8-way `ldmatrix` bank conflict. The blog reports ~2× and ~50% of cuBLAS.

**Kernel 4: prefetch one block tile ahead (register staging).** The diff vs kernel 3:
- L85–89 `static_assert` BM=256, BN=128, BK=64, 256 threads. They declare `float4 A_gmem_cache_reg[8]` and `B_gmem_cache_reg[4]`.
- L91–98 is a prologue: copy tile 0 into SMEM, then `__syncthreads`.
- The loop becomes `for block_k = 1..num_block_tiles_k` (L101). Each iteration:
  - `tileMemcpyLoad` of tile k into registers (L103–109).
  - `__syncthreads` (L111).
  - Compute from SMEM.
  - `__syncthreads` (L187).
  - `tileMemcpySwizzleStore` registers → SMEM (L190–194).
- Buys: global latency overlaps with the HMMA work. Costs registers (104 → 166 per the blog). The blog reports ~70% of cuBLAS.

**Kernel 5: retiled + hand-unrolled `ldmatrix` with XOR stepping** (`kernel5.cu`)
- Tiles (L416–448): BM=BN=256, BK=32. Warps 2 (M) × 4 (N), so **WM=128, WN=64**. SMEM = 32 KB.
- `mma_tiles_per_warp_k = 4` is hard-coded (L270). Each warp covers the full BK=32 per block tile.
- Registers: `acc[8][8][2]` = 128 accumulator regs per thread, `A[8][4][2]` = 64, `B[4][8]` = 32.
- `ldmatrix_a` (L9–158): 16 × `ldmatrix.x4`, each loading two 16×8 A tiles.
  - The swizzled base is computed once (L19–22).
  - M steps are `+32*stride` adds. K steps are `src_addr ^= 0b10000`, `^= 0b110000`, `^= 0b10000`, i.e. XOR constants instead of recomputing the swizzle.
- `ldmatrix_b` (L160–237): 8 × `ldmatrix.x4.trans`. The base swizzle is `^ ((off & 0b11100000000) >> 5)`. N halves via `src_addr ^ 0b1000000`, K steps via `+= 8*stride`.
- Main loop (L328–376):
  - `__syncthreads`, then the register prefetch of tile k.
  - `ldmatrix_a/b`, then a fully `#pragma unroll`ed mma triple loop.
  - `__syncthreads` (L369), then the swizzled store.
  - A uses `tileMemcpySwizzleA`/`StoreA` for the 32-wide tile.
- Buys: 1.2× over kernel 4 and 86.7% of cuBLAS (blog). The instruction count drops below cuBLAS.

**Kernel 6: double-buffered SMEM.** The diff vs kernel 5:
- L285: `BUFFER_SIZE = BM*BK + BK*BN`.
- L329: `int offset_direction = 1`.
- L342–343 recompute the warp tile pointers each iteration.
- **The `__syncthreads()` before the store is deleted.**
- L377–379, before the store: `A_block_smem += BUFFER_SIZE*offset_direction` (same for B), then `offset_direction = -offset_direction`. Tile k+1 is written into the *other* buffer while slower warps may still read the current one.
- Launch SMEM is doubled (L453): `... * 2 * sizeof(half)` = 64 KB.
- Correctness argument: the single `__syncthreads` at the loop top (L333) separates "everyone finished reading buffer X" from "someone writes X again", two iterations later.
- Buys: a small gain (the blog gives CPI 14, ~2.6 cycles of sync stall). The README says ~95% of cuBLAS.

## Numbers worth remembering
T4 GFLOP/s, **read off `3-GEMM/hgemm_performance.png`** (approximate, ±~500):

| M=N=K | k1 | k2 | k3 | k4 | k5 | k6 | cuBLAS |
|---|---|---|---|---|---|---|---|
| 1024 | ~3.4k | ~9.7k | ~17.8k | ~20.2k | ~16.3k | ~16.3k | ~25.5k |
| 2048 | ~3.3k | ~11.4k | ~21.8k | ~26.1k | ~32.8k | ~33.6k | ~39.3k |
| 4096 | ~4.0k | ~14.0k | ~28.7k | ~33.8k | ~41.8k | ~43.1k | ~49.4k |
| 8192 | ~4.2k | ~13.1k | ~27.7k | ~33.7k | ~42.3k | ~46.9k | ~48.8k |

- At **1024³, kernels 5/6 (256×256 tiles) are slower than kernel 4 (256×128).**
  - Inference: 1024/256 squared = 16 CTAs on a 40-SM T4, so most SMs idle.
  - This is the same failure mode a skinny GEMM hits when tiles are big.

## Code patterns
Swizzle on store and on load (kernel 3, applied in two places with the same function):
```cuda
dst_index = dst_index ^ ((dst_index & SWIZZLE_MASK) >> SWIZZLE_BITS);          // device_utils.cuh L146 (float4 index)
thread_offset_bytes ^= ((thread_offset_bytes & SWIZZLE_MASK_A) >> SWIZZLE_BITS_A); // kernel3.cu L112 (byte offset)
```
The double-buffer flip (kernel 6, L377–379):
```cuda
A_block_smem = A_block_smem + BUFFER_SIZE * offset_direction; // ping ↔ pong
B_block_smem = B_block_smem + BUFFER_SIZE * offset_direction;
offset_direction = -1 * offset_direction;
```

## Pitfalls
- The inline-asm operand lists in `kernel1.cu` L147–148 (and the same in kernels 2–6, e.g. `kernel5.cu` L361–362) have **no comma** between `"r"(B_register[...])` and `"r"(acc_register[...][0])`. The code apparently built for the author, but don't copy it. Use the `mma_sync_m16n8k8` wrapper (device_utils L485–508) as the template. Not compiled here.
- Kernel 5's `static_assert` message says "must be 4" while it asserts 8 (L15). The `// WM = 128, WN = 64, WK = 8` comment in launch_5 conflicts with `mma_tiles_per_warp_k = 4` (i.e. WK = 32).
- FP16 accumulators; α/β applied in half.
- The 100 ms sleep after each timed launch means every launch runs from a cold, idle GPU. Not comparable to CUDA-graph replay timing → [[benchmarking-kernels]].
- `-dlcm=cg` disables L1 caching of global loads for every kernel.

## Relevance to the serving track
- **Reusable as-is (inference):**
  - `cvta_to_shared_u32`.
  - The `ldmatrix`/`mma` wrapper style.
  - `tileMemcpyLoad`/`Store` as a register-staged prefetch. Useful in a [[triton]]-free CUDA kernel on Hopper when you don't want `cp.async` group bookkeeping. On sm_90 prefer [[cp-async]] or [[tma]].
  - The XOR-constant stepping trick for swizzled `ldmatrix` addresses.
- **Small-m changes needed (inference):**
  - Switch to `m16n8k16.row.col.f32.bf16.bf16.f32` with FP32 accumulate.
  - Drop `.trans`: W `[N,K]` is K-contiguous.
  - Tile M to at most 16–32 (or swap AB so tokens sit on mma-N=8).
  - Shrink BN so `N/BN × splitK ≥ #SMs`. Add K-tail/M-tail predication.
  - Write a coalesced, split-K-aware epilogue.
  - The 256×256 tiles here are the opposite of what m ≤ 256 needs.
- The 1024³ regression (big tiles + few CTAs) is the cheapest demonstration in this vault of why **SM fill beats per-CTA efficiency** for small problems → [[gemv-and-skinny-gemm]].

## Related
[[alexarmbr-tensor-core-hgemm]] · [[mma-ptx]] · [[gemm-tiling-hierarchy]] · [[cp-async]] · [[swizzling]] · [[smem-bank-conflicts]] · [[split-k-and-stream-k]] · [[gemv-and-skinny-gemm]] · [[hidden-treasures-llms]] · [[benchmarking-kernels]]
