---
title: Shared Memory Swizzling (XOR swizzle, 32/64/128B modes)
type: concept
tags: [gpu, hopper, swizzle, smem, tma, wgmma, bank-conflicts]
source_url:
  - https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
  - https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/
  - https://github.com/simveit/effective_transpose
  - raw/hidden-treasures-llms/swizzle.py
---
# Shared Memory Swizzling (XOR swizzle, 32/64/128B modes)

**TL;DR**
- Swizzling permutes 16-byte chunks within each row of a smem tile by XOR-ing the chunk index with low bits of the row index, so that column-wise (or core-matrix) accesses spread across all banks with zero padding.
- TMA supports 4 modes (`CUtensorMapSwizzle`): NONE, 32B, 64B, 128B. Granularity is fixed at 16 B. 128B: `smem[y][x] <-> smem[y][((y+offset)%8) ^ x]` where x,y index 16-B chunks / 128-B rows (CUDA Programming Guide, Table 27).
- Repeat period: 128B → 1024 B, 64B → 512 B, 32B → 256 B. Box inner dim must be ≤ swizzle span (128/64/32 B). Smem address 128 B aligned; align buffers to the period "for simplicity"; global 128 B aligned for swizzled maps.
- wgmma smem descriptors (see [[wgmma]]) encode the same swizzle modes; the TMA-written layout and the descriptor's layout must agree. In CuTe this is `Swizzle<3,4,3>` for 128B (see [[cute-layouts]]).
- Veitner: adding 128B swizzle to a TMA transpose took 875 → 1252 GB/s.

## Key ideas
- **Why XOR**: a 128-B row covers all 32 banks once (8 chunks × 4 banks). Rows of a tile with pitch 128 B start in the same bank, so reading "chunk c of rows 0..7" hits the same 4 banks 8 times. XOR-ing c with (row mod 8) maps those 8 reads to 8 distinct chunk slots → all 32 banks. XOR is a bijection on each row, so no data collides and no space is wasted.
- **Modes** (official, Table 28):
  | Mode | Span | Max box inner dim | Period | Smem align | Gmem align |
  |---|---|---|---|---|---|
  | 128B | 128 B | 128 B | 1024 B | 128 B | 128 B |
  | 64B | 64 B | 64 B | 512 B | 128 B | 128 B |
  | 32B | 32 B | 32 B | 256 B | 128 B | 128 B |
  | NONE | – | (16 B listed) | – | – | – |
  The 64B/32B modes XOR with (row % 4) / (row % 2) and only permute within 64/32-B spans – fewer distinct bank groups, so only fully conflict-free when the access pattern spans ≤ 4 or ≤ 2 rows.
- **Address phase**: the pattern is keyed to *absolute smem address bits* (the guide's offset is `(smem_ptr/128) % 8` for 128B), so a buffer not aligned to 1024 B starts mid-pattern. Simplest rule: `alignas(1024)` for 128B-swizzled buffers (Veitner does this).
- **CuTe notation** (inference/common knowledge, see [[cute-layouts]]): `Swizzle<B,M,S>` XORs bits [M+S, M+S+B) into bits [M, M+B) of the element/byte offset. 128B = `Swizzle<3,4,3>` on byte offsets (16-B units, 8 chunks, rows of 128 B); 64B = `Swizzle<2,4,3>`; 32B = `Swizzle<1,4,3>`.
- **wgmma/TMA pairing** (inference): for BF16 K-major operands the canonical choice is a 64-element (128 B) inner box with SWIZZLE_128B, giving 8-row × 128-B "swizzle atoms" that the wgmma descriptor walks with a fixed stride byte offset of 1024 B.

## Numbers worth remembering
- 16 B swizzle granularity; periods 256/512/1024 B; max inner box 32/64/128 B (CUDA guide).
- TMA tensor copy: gmem addr 16 B aligned, strides multiple of 16 B, smem 128 B aligned, size multiple of 16 B (CUDA guide Table 26).
- Veitner FP32 transpose: unswizzled 875 GB/s → 128B swizzle 1252 GB/s.

## Code patterns
Official index relation (CUDA guide), x = 16-B chunk index in row, y = 128-B row:
```cpp
int offset = (reinterpret_cast<uintptr_t>(smem_ptr) / 128) % 8; // 0 if 1024B-aligned
// logical smem[y][x]  lives at physical smem[y][((y + offset) % 8) ^ x]
```
Veitner's element-level version (FP32, 32-float = 128 B rows):
```cpp
int i16 = (row * 32 + col) * 4 >> 4;  // 16B chunk index from element index
int y16 = i16 >> 3, x16 = i16 & 7;    // row-of-128B, chunk-in-row
int phys_col = (((y16 ^ x16) * 4) & 31) + (col & 3); // swizzled chunk, same float inside chunk
```
Generic byte-offset form (inference; equals the above when aligned):
```cpp
__device__ uint32_t swz128(uint32_t byte_off) { return byte_off ^ ((byte_off >> 3) & 0x70); }
// bits [7,10) (row mod 8) XORed into bits [4,7) (16B chunk index)
```
TMA map: `cuTensorMapEncodeTiled(..., CU_TENSOR_MAP_SWIZZLE_128B, ...)` with `box[0]*elem_size <= 128`.

## swizzle.py walkthrough (raw/hidden-treasures-llms/swizzle.py)
Intent (docstring): map a byte address that starts a 16-B core-matrix chunk to one of **8 "banks" of 16 B** (i.e. 8 bank-groups of 4 physical banks = one 128-B row), with Hopper-style XOR selector `s ∈ {0..3}` (s=0 none, 1 ≈ 32B, 2 ≈ 64B, 3 ≈ 128B).
```python
word = addr >> 2                               # byte -> 4-byte word index
folded_bits = (word >> 3) & ((1 << s) - 1)     # "row" bits above the 8-slot field, keep s of them
return (word ^ folded_bits) & 0x07             # XOR row bits into slot index, 8 slots
```
`group_by_bank8(limit, s)` buckets addresses 0..limit (step 16 B) by bank; plotting code is commented out; the script's live loop prints banks for addr 0..0x100 step 4 with s=0.

**Caveat found when re-implementing it (inference, checked by running the logic):** the code shifts by 2 (4-B words), so its 8 slots are 4-B words and a "row" is 32 B – not the 16-B-slot / 128-B-row model the docstring describes. With its own 16-B step, s=0 yields banks `0,4,0,4,…`, s=1 `0,4,1,5,…`, and s=2/s=3 `0,4,1,5,2,6,3,7,…` (all 8 slots, but via the 4-B/32-B model, not the 16-B/128-B one). Changing `addr >> 2` to `addr >> 4` gives the intended model, and then the slot table for rows r=0..7, chunks k=0..7 is `k ^ (r & (2^s-1))`:
- s=0: every row `0..7` identical → column of chunk k across 8 rows = 8-way conflict.
- s=1 (32B-like): rows alternate `01234567` / `10325476` → 2 distinct slots per column.
- s=2 (64B-like): 4-row cycle → 4 distinct.
- s=3 (128B): rows are `k ^ r` → every column of 8 rows hits 8 distinct slots = all 32 banks, conflict-free. This is exactly the CUDA guide's `((y+offset)%8) ^ x` with offset 0.

## Pitfalls
- Swizzle mode on the tensor map must match the consumer (wgmma descriptor layout type, ldmatrix address math, or manual index function). Mismatch = silently wrong results, not a crash.
- Box inner dim > swizzle span is rejected by `cuTensorMapEncodeTiled` (128B mode: ≤ 128 B → ≤ 64 BF16 / 32 FP32 per row; tile wider K as multiple boxes).
- Buffer not aligned to the period → manual index functions assuming offset 0 are wrong.
- Swizzle doesn't help bandwidth if the kernel is HBM-bound; it only removes smem serialization.

## Relevance to the serving track
- (inference) For `muse_bf16_proj` with a TMA+wgmma design, W (n×k, K-major since out = x @ W.T) maps naturally to a 2D tensor map with box `{64 (k), BN (n)}` BF16, SWIZZLE_128B; x similarly `{64, m}`. This is the layout wgmma's B (and A) smem descriptors want.
- (inference) For a non-tensor-core GEMV path (m=1..8) using LDG.128 directly into registers, no smem/swizzle is needed for W at all – only x goes through smem (broadcast).
- (inference) The FP32 router as BF16 hi+lo: the hi and lo planes are separate tensors with identical swizzled layout; two TMA loads per stage, same descriptors.

## Related
[[smem-bank-conflicts]] · [[tma]] · [[wgmma]] · [[cute-layouts]] · [[mma-ptx]] · [[veitner-hopper-transpose]] · [[feldmann-smem-microbenchmarks]] · [[yangyifan-cute-tma]] · [[mbarrier]]
