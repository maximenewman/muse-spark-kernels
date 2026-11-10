---
title: "mma.sync PTX: shapes, bf16/fp16 fragment layouts, ldmatrix, WMMA vs MMA"
type: concept
tags: [gpu, tensor-cores, mma-ptx, ldmatrix, wmma, bf16, fragments, ampere, hopper]
source_url:
  - https://docs.nvidia.com/cuda/parallel-thread-execution/index.html   # §9.7.16.5.8 (m16n8k16 fragments), §9.7.16.5.14 (mma), §9.7.16.5.15 (ldmatrix), §9.7.16.4.1 (wmma fragments)
  - https://github.com/Bruce-Lee-LY/cuda_hgemm
  - https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
---
# mma.sync PTX

**TL;DR**
- **The BF16 instruction:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. BF16 mma exists **only with f32 accumulators**, in shapes m16n8k8 and m16n8k16, on sm_80+. FP16 allows `.f16` or `.f32` C/D.
- **Per thread for m16n8k16 bf16:** A = 4 × b32 regs (8 bf16), B = 2 × b32 (4 bf16), C/D = 4 × f32. With `g = lane>>2` and `t = lane%4`, thread (g,t) owns C rows {g, g+8} × cols {2t, 2t+1}.
- **Loading:**
  - `ldmatrix.sync.aligned.m8n8.x4.shared.b16` gives a full A fragment (16×16).
  - `.x2` gives a B fragment (16k × 8n) when B is stored `[N][K]`, i.e. K-contiguous. **That is exactly `W[N,K]` for `x @ W.T`, so no `.trans` is needed.**
  - Use `.trans` only when the tile in SMEM is stored the other way (e.g. B as `[K][N]`).
- **WMMA** (`nvcuda::wmma`, m16n16k16 / m32n8k16 / m8n32k16) has an **unspecified fragment layout**, so element-wise epilogues need a store round-trip. MMA PTX gives exact ownership, which allows m16n8 granularity, register double-buffering and custom swizzles.
- On H100/H200, `mma.sync` still works. Peak-rate dense GEMM on Hopper needs [[wgmma]]. At m ≤ 256 the kernel is HBM-bound, so `mma.sync` is usually adequate (inference).

## Key ideas
### Shapes and types (PTX ISA §9.7.16.5.14 syntax block)
| Inputs | Shapes | C/D types | Min arch |
|---|---|---|---|
| `.f16` | m8n8k4, m16n8k8, **m16n8k16** | `.f16` or `.f32` | m16n8k8: sm_75; m16n8k16: sm_80 |
| `.bf16` | m16n8k8, **m16n8k16** | **`.f32` only** (`mma...f32.bf16.bf16.f32`) | sm_80 |
| `.tf32` | m16n8k4, m16n8k8 | `.f32` | sm_80 |
| `.e4m3/.e5m2` | m16n8k16, m16n8k32 | `.f16`/`.f32` | sm_89 |

- `.row.col` is the only layout for m16n8kX: A is row-major (M×K, K contiguous) and B is "col" (K×N with K contiguous per column, i.e. stored `[N][K]`). Layout choice is about how the **fragment** is indexed. Your SMEM/global layout can be anything, as long as `ldmatrix` (or manual loads) produce the fragment.
- `.sync.aligned`: all 32 lanes must execute the same instruction convergently.
- The ISA notes say m16n8k8/m16n8k16 bf16/tf32 "require at least 64 registers for compilation" (I read this as a minimum per-thread register budget).

### Fragment layouts for m16n8k16, .f16/.bf16 (PTX §9.7.16.5.8)
`groupID g = laneid >> 2`, `threadID_in_group t = laneid % 4`.

| Operand | Regs | Element i → (row, col) |
|---|---|---|
| A (16×16) | 4 × .b32 = a0..a7 | row = g for i ∈ {0,1,4,5}, else g+8. col = 2t + (i&1) for i < 4, else 2t + (i&1) + 8 |
| B (16×8) | 2 × .b32 = b0..b3 | row(k) = 2t + (i&1) for i < 2, else 2t + (i&1) + 8. col(n) = g |
| C/D (16×8) f32 | 4 × .f32 = c0..c3 | row = g for i < 2, else g+8. col = 2t + (i&1) |

Reading the table:
- Register order is `a0a1` = (g, k 2t..2t+1), `a2a3` = (g+8, k 2t..), `a4a5` = (g, k 2t+8..), `a6a7` = (g+8, k 2t+8..). This is exactly what `ldmatrix.x4` produces when lanes 0–15 point at rows 0–15, k-offset 0, and lanes 16–31 point at rows 0–15, k-offset 8 (matrix order: rows 0–7/k0, rows 8–15/k0, rows 0–7/k8, rows 8–15/k8).
- B: thread (g,t) holds column n = g, k = {2t, 2t+1, 2t+8, 2t+9}. Each 32-bit reg holds 2 **consecutive k** of one **n**. So B wants k-contiguous storage per n, i.e. `W[n][k]`.
- C: each thread holds 2 adjacent columns in 2 rows. A row of 8 outputs is spread across the 4 threads of a group. A coalesced epilogue usually goes through SMEM, or uses `stmatrix` (sm_90+).

m16n8k8 (used by [[alexarmbr-tensor-core-hgemm]]): A = 2 regs, B = 1 reg, f16 C/D = 2 regs (from the repo's wrapper signatures).

### ldmatrix (PTX §9.7.16.5.15)
- `ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}.shared.b16 r, [p]` loads 1/2/4 8×8 b16 matrices.
- **Each lane supplies one row address:**
  - lanes 0–7 → rows of matrix 0
  - lanes 8–15 → matrix 1
  - lanes 16–23 → matrix 2
  - lanes 24–31 → matrix 3
- Each row is 16 B and must be 16 B aligned. Rows need not be contiguous, which is why padding and swizzles work.
- Result: thread `l` gets the 2 elements at (row l/4, cols 2(l%4)..+1) of each matrix, one b32 register per matrix. That is precisely the A/B/C fragment sub-layout above.
- `.trans` delivers the transposed 8×8, used when the tile is stored in the other orientation (e.g. B row-major `[K][N]` in [[hidden-treasures-3-gemm-code]]).
- On sm_75 or below, all lanes must hold valid addresses even for `.x1`/`.x2`. Copying the lower lanes' addresses into the upper lanes works.
- **Bank conflicts:** the 8 row addresses of one 8×8 matrix are serviced together. If they map to the same 4 banks (row stride a multiple of 128 B), you get an 8-way conflict → [[swizzling]], [[smem-bank-conflicts]].
- `ldmatrix` is in the generic proxy and is a weak memory op. A `__syncthreads()` (or a barrier) is still needed after cooperative SMEM writes.

### WMMA vs MMA
| | WMMA (`nvcuda::wmma`) | MMA PTX (`mma.sync`) |
|---|---|---|
| Shapes (f16/bf16) | m16n16k16, m32n8k16, m8n32k16 | m16n8k8, m16n8k16 |
| Fragment layout | "unspecified and target architecture dependent" (PTX §9.7.16.4.1) | Fully specified (table above) |
| Loads | `load_matrix_sync(frag, ptr, ldm)` hides the smem→reg load | You issue `ldmatrix` or plain LDS/LDG yourself |
| Epilogue | `store_matrix_sync` then re-read | Per-element math on known (row, col), direct stores |
| Bank-conflict fix used in [[bruce-lee-mma-ptx]] | Padding (+8 halves per row) | Permuted/XOR column offsets |

alexarmbr chose PTX "because WMMA hides the shared-to-register loads", which he found critical for performance ([[alexarmbr-tensor-core-hgemm]]).

## Code patterns
```cuda
// BF16 m16n8k16, FP32 accumulate. A: 4 regs, B: 2 regs, C/D: 4 f32 (can alias C and D).
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
  : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
  : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

// A fragment (16 rows x 16 k) from smem tile As[rows][lda] (k contiguous):
uint32_t a_addr = __cvta_generic_to_shared(&As[lane % 16][(lane / 16) * 8]);
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));
// B fragment (16 k x 8 n) from smem Ws[n][ldw] (k contiguous, i.e. W rows):
uint32_t b_addr = __cvta_generic_to_shared(&Ws[lane % 8][((lane / 8) % 2) * 8]);
asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
             : "=r"(b[0]), "=r"(b[1]) : "r"(b_addr));  // lanes 16-31 addresses ignored for .x2 on sm_80+
```
- The bf16 inline-asm form above is my adaptation of [[bruce-lee-mma-ptx]]'s `HMMA16816` macro (f16→bf16, f16 acc → f32 acc), following the PTX syntax line. **Not compiled here.**
- Dropping `ldmatrix`: because a dot product is invariant under any permutation of k applied to both operands, a thread can load 16 B (8 consecutive k) of one W row straight from global into registers. It then treats them as {a0a1, a4a5} of two consecutive k16 steps, provided x/B is loaded with the *same* k permutation. That's a register-direct weight stream with 128-bit coalesced LDGs and no SMEM (inference; check correctness against a reference).

## Pitfalls
- Using the `.f16` accumulator form with bf16 inputs is invalid PTX. bf16 needs `.f32.bf16.bf16.f32`.
- Operand lists in inline asm must be comma-separated. [[hidden-treasures-3-gemm-code]] kernels have a missing comma; don't copy it.
- `ldmatrix` addresses are 32-bit **shared-window** addresses (`__cvta_generic_to_shared` / `cvta.to.shared`).
- WMMA fragments from code compiled for different SM targets aren't interchangeable (PTX note).
- Putting m < 16 tokens in the mma M slot wastes 16−m rows of tensor work and A-fragment registers. Swap AB (tokens → N=8) for m ≤ 8 (inference).

## Relevance to the serving track
- `muse_bf16_proj` (`x @ W.T`, W `[N,K]` bf16) has two operand mappings (inference):
  1. **x as A (M=16 per mma), W as B (N=8 per mma).** W loads via `ldmatrix.x2` (no trans), x via `ldmatrix.x4`. At m=1 that's 15/16 waste on the A side; it's harmless when memory-bound, but registers and shuffles still scale with the 16.
  2. **Swap AB: W as A (16 W-rows per mma), x as B (8 tokens per mma).** m=8 is a perfect fit, m=1 wastes 7/8 of N (harmless). Output comes out transposed (thread owns 2 tokens × 2 W-rows), so the epilogue writes `out[token][n]` with the transposed index. This is what the user's Triton kernel does ([[gemv-and-skinny-gemm]]).
- **FP32 router via BF16 hi+lo:** `x_hi + x_lo` and `W_hi + W_lo` split, then 3 mmas (hi·hi + hi·lo + lo·hi) into the same f32 accumulators. Native f32 accumulate makes this cheap (inference; the user's kernel-track note says router runs as hi+lo at m ≥ 2).
- At m ≤ 256 on H200 the tensor-core rate isn't the limiter. Choose the mapping that minimizes **bytes moved and instructions per W byte**, not FLOP efficiency (inference).

## Related
[[bruce-lee-mma-ptx]] · [[alexarmbr-tensor-core-hgemm]] · [[hidden-treasures-3-gemm-code]] · [[gemm-tiling-hierarchy]] · [[cp-async]] · [[wgmma]] · [[swizzling]] · [[smem-bank-conflicts]] · [[gemv-and-skinny-gemm]] · [[tensor-core-generations]]
