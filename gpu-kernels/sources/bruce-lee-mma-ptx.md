---
title: "Bruce-Lee-LY cuda_hgemm (WMMA vs MMA PTX) + 'Getting started with MMA PTX'"
type: source
tags: [gpu, gemm, hgemm, wmma, mma-ptx, ldmatrix, cp-async, multistage, ampere]
source_url:
  - https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master/src/wmma
  - https://github.com/Bruce-Lee-LY/cuda_hgemm/tree/master/src/mma
  - https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d   # NOT READ: HTTP 403
---
# Bruce-Lee-LY cuda_hgemm: WMMA vs MMA PTX

**TL;DR**
- An Ampere (sm_80/sm_86) FP16 HGEMM ladder written twice: once with the **WMMA API** (`src/wmma/*.cu`, m16n16k16 fragments) and once with **raw MMA PTX** (`src/mma/*.cu`, `ldmatrix` + `mma.sync.m16n8k16`). Both use `cp.async` multistage pipelines.
- README claim: "between 256 to 16384 dimensions is not less than 95% of the performance of cublas", on RTX 3090 / RTX A6000 with CUDA 11.3 (numbers only in charts).
- Shared skeleton: block 256×128, warp 64×64, 8 warps, `CHUNK_K = 2` (32 halves of K per stage), 16 B per thread copies, and a block-index "swizzle" for L2 (`BLOCK_STRIDE 16`).
- **Bank conflicts:** WMMA versions **pad** SMEM rows by 8 halves (`AB_SMEM_STRIDE 40`). MMA versions use a **permuted (XOR-like) column offset** at stride 32 with no padding.
- **The B layout matches our problem:** B is stored `[N][K]` (K contiguous), so `B` fragments load with plain `ldmatrix.x2`, no `.trans`. That's the same layout as `W` in `x @ W.T`.
- **The Medium article returned HTTP 403** (also via medium.com/@...). Its content is not summarized here; everything below comes from the repo source and README.

## Key ideas: kernel ladder (from file names, defines and code)
| WMMA file | MMA file | Optimization added |
|---|---|---|
| `wmma_naive.cu` | `mma_naive.cu` | One warp per 16×16 (WMMA) or 16×8 (MMA) output tile. Global → fragment directly (WMMA) or a tiny SMEM hop (MMA) |
| `wmma_base.cu` | `mma_base.cu` | Block 256×128, warp 64×64, SMEM staging with `int4` copies, `AB_SMEM_STRIDE 32` |
| `wmma_padding.cu` | `mma_permuted.cu` | Bank-conflict removal: +8 halves of padding (WMMA) vs a permuted column offset (MMA) |
| `wmma_async.cu` | `mma_async.cu` | `cp.async.cg ... 16` with `commit_group` and `wait_group 0`. Still single-buffered |
| `wmma_async_pg2s.cu` | `mma_async_pg2s.cu` | Double-buffer **global→SMEM** ("prefetch g2s") |
| `wmma_async_pg2s_ps2r.cu` | `mma_async_pg2s_ps2r.cu` | Also double-buffers **SMEM→registers** (two fragment sets `[2]`) |
| `wmma_async_stage2/3.cu` | `mma_async_stage2/3/4.cu` | `K_STAGE`-deep ring of SMEM buffers. Keeps `K_STAGE-1` groups in flight with `cp.async.wait_group(K_STAGE-2)` |

The README's optimization list adds "Register Reuse: calculating as 'Right Left Right Left' for the internal tile of warp". That's the serpentine `j_s = (i % 2) ? (WARP_ROW_TILES - j - 1) : j` in `mma_async_stage3.cu` L259/L376, which reuses the last B fragment across consecutive mmas.

### WMMA vs MMA (as visible in the code)
- **WMMA:** `wmma::fragment<matrix_a,16,16,16,half,row_major>`, `load_matrix_sync(frag, ptr, ldm)`, `mma_sync`, `store_matrix_sync`. It needs no lane-address math. But the **fragment-to-element mapping is opaque**: PTX says it is "unspecified and is target architecture dependent". The epilogue therefore has to go through `store_matrix_sync` (to SMEM/global), and you can't fuse per-element math on known (row, col) positions.
- **MMA PTX:** you compute each lane's `ldmatrix` row address yourself (`lane % 16`, `(lane / 16) * 8` for A; `lane % 8`, `((lane / 8) % 2) * 8` for B). The fragment layout is fully specified (see [[mma-ptx]]). The finer m16n8 granularity and the explicit registers enable register double-buffering, serpentine order, and custom swizzles.
- WMMA also offers m8n32k16 and m32n8k16 for f16/bf16 (PTX ISA). These are the only WMMA shapes with a dimension < 16 (inference: m8n32k16 is the WMMA shape that fits M=8 tokens without padding).

## Numbers worth remembering
- Tiles: 256×128 block, 64×64 warp, 8 warps, 256 threads. `WARP_COL_TILES 4 × WARP_ROW_TILES 8` mma tiles per warp (MMA), i.e. `RC[4][8][2]` = 64 f16x2 accumulator regs.
- `mma_async_stage3`: SMEM = `max((256+128)·32·2 B·3, 256·128·2 B)` = 73,728 B for the A/B ring (the C staging buffer reuses it). Set via `cudaFuncAttributeMaxDynamicSharedMemorySize`.
- Claimed: ≥95% of cuBLAS from 256 to 16384 (RTX 3090 / A6000, CUDA 11.3). No small-M data.

## Code patterns
PTX macros (`src/common/ptx.h`, verbatim):
```cuda
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))
```
The minimal MMA tile (`mma_naive.cu` L36–55; one warp, A `[M][K]`, B `[N][K]`):
```cuda
*((int4*)(&A_smem[lane_id / 2][0]) + lane_id % 2) =                  // 16 rows × 16 halves = 32 lanes × 16 B
    *((int4*)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);
if (lane_id < MMA_N * 2)                                              // B: 8 rows (n) × 16 halves (k)
    *((int4*)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
        *((int4*)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
__syncthreads();
uint32_t A_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]); // lanes 0-15: rows, k 0-7; 16-31: k 8-15
LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_addr);                     // a0..a7 of m16n8k16
uint32_t B_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]); // n rows; lanes 8-15: k 8-15
LDMATRIX_X2(RB[0], RB[1], B_addr);                                    // b0..b3, no .trans since B is [N][K]
HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
```
Accumulator write-out (L60–61): `C_smem[lane/4][...] + lane%4` ← `RC[0]`, and row `lane/4 + 8` ← `RC[1]`. That's the C-fragment layout from [[mma-ptx]].

`mma_async_stage3.cu` pipeline skeleton (L147–193, L226–342):
```cuda
// prologue: issue stages 0 and 1, commit each, wait until only 1 group pending → stage 0 ready
CP_ASYNC_COMMIT_GROUP(); ... CP_ASYNC_COMMIT_GROUP(); CP_ASYNC_WAIT_GROUP(1); __syncthreads();
for (tile_k = CHUNK_K*(K_STAGE-1); tile_k < K_tiles; tile_k += CHUNK_K) {
  // ldmatrix k-step 1 of current stage into reg buffer (ps2r double buffer), HMMA on reg buffer from k-step 0
  // issue cp.async for stage (tile_k) into ring slot smem_store_idx = (idx+1) % K_STAGE
  CP_ASYNC_COMMIT_GROUP(); CP_ASYNC_WAIT_GROUP(1); __syncthreads();
  // advance smem_load_idx, ldmatrix k-step 0 of next stage, HMMA on previous regs
}
// drain: CP_ASYNC_WAIT_GROUP(0) before the last stage, then finish remaining k-steps
```
Permuted SMEM column for bank-conflict-free `ldmatrix` without padding (L205–207):
```cuda
&smem[A_smem_idx + lane_id % 16][((lane_id / 16) * 8 + (lane_id % 16 % (PERMUTED_COLS * SMEM_BANK_ROWS)) / SMEM_BANK_ROWS * PERMUTED_OFFSET) % AB_SMEM_STRIDE]
// AB_SMEM_STRIDE 32 halves = 64 B → 2 rows per 128 B bank line (SMEM_BANK_ROWS 2); rotate 16 B chunk by row-pair
```

## Pitfalls
- `.f16` accumulators everywhere. For bf16 use `m16n8k16.row.col.f32.bf16.bf16.f32`, where C/D are 4 f32 regs (not 2 packed).
- The copy loops assume full tiles. `div_ceil` grids plus an early return on `block_tile_i >= M_tiles` handle whole-tile overrun only, not partial K/M tails.
- The macro uses `.x4.m8n8` qualifier order (`ldmatrix.sync.aligned.x4.m8n8...`). The PTX ISA syntax line is `.shape.num` (`.m8n8.x4`). The repo builds with it, but when writing new code, follow the ISA order.
- The L2 `::128B` prefetch qualifier needs CUDA ≥ 11.4, handled by the `#if` in ptx.h.

## Relevance to the serving track
- **Transfers directly (inference):**
  - The `[N][K]` B layout, `LDMATRIX_X2` without trans, and `HMMA16816` are exactly the operand path for `x @ W.T` with W as B and x as A. Change the type string to `.f32.bf16.bf16.f32` and use 4 f32 accumulators.
  - If you **swap AB** (W as A, tokens as B/N=8), W uses `LDMATRIX_X4` per 16 rows × 16 k. The m=1–8 tokens fill mma-N=8, with zero waste at m=8.
  - The `cp.async` multistage ring (stage 3–4) is the standard way to keep enough W bytes in flight per SM on sm_80/sm_90 without [[tma]] → [[cp-async]].
- **Does not transfer:**
  - The 256×128 block tile. At m ≤ 256 the M tile collapses to 16–64 and BN must shrink to fill 132 SMs.
  - The L2 block swizzle (`BLOCK_STRIDE`), useless when every W byte is read once.
  - WMMA in general: opaque fragments, and the 16×16×16 granularity forces padding m up to 16 (or 8 with m8n32k16) (inference).

## Related
[[mma-ptx]] · [[cp-async]] · [[gemm-tiling-hierarchy]] · [[alexarmbr-tensor-core-hgemm]] · [[hidden-treasures-3-gemm-code]] · [[smem-bank-conflicts]] · [[swizzling]] · [[gemv-and-skinny-gemm]] · [[hidden-treasures-llms]]
