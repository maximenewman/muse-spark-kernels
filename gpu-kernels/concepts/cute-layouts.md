---
title: CuTe Layouts
type: concept
tags: [gpu, cute, cutlass, layout, tiling, tma]
source_url:
  - https://yang-yifan.github.io/blogs/cute_layout/cute_layout.html
  - https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html
  - https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html
---
# CuTe Layouts

**TL;DR**
- `Layout = Shape : Stride`: a function from (possibly nested) coordinates → linear index. Default stride is **column-major (LayoutLeft)**; write `make_stride(K, _1{})` for PyTorch row-major.
- Any layout is also a 1D-coord → index map (1D coord = column-major linearization of the shape); size = domain size, cosize = codomain size.
- Algebra used daily: `coalesce` (flatten), `blocked_product`/`logical_product` (repeat a tile), `local_tile`/`inner_partition` (slice a CTA tile), `partition_S/D` (reshape for a copy atom).
- Static ints (`Int<N>{}`, `_1{}`) for anything in RMEM/SMEM/TMEM and for TMA/MMA tile shapes; dynamic ints only for gmem problem sizes.
- TMA uses **coordinate tensors** (ArithTuple from `get_tma_tensor`) manipulated with the same layout algebra as the smem tensor.

## Key ideas
- **Shape/stride**: stride[i] = index increment when coord i increments by 1. `(M,N,K):(1,M,MN)` col-major; `(M,N,K):(NK,K,1)` row-major. Nesting: `((2,2),2):((4,1),2)` is a 4×2 matrix indexed `((m,n),k)`.
- **Coalesce** rules: drop static size-1 modes; merge `(s0,s1):(d0,s0*d0) → s0*s1:d0`. By-mode: `coalesce(L, Step<_1,_1,_1>{})` keeps rank = rank of Step.
- **Blocked product**: tile `(M0,N0):(1,M0)` × row-major tiler `(M1,N1):(N1,1)` = `((M0,M1),(N0,N1)) : ((1,N1·M0·N0),(M0,M0·N0))` → coalesces to `((M0,M1),N0·N1):((1,N1·M0·N0),M0)`. It is two 1D logical products done sequentially because the M1 stride depends on N0,N1.
- **local_tile(T, Tile<BM,BK>, coord)** ≡ numpy slice of tile `coord`; the tiler is a shape tuple (modes partitioned independently). Not required to invert the product that built T.
- **partition_S / partition_D** (copy atoms): reshape a tile `[BM,BK]` → `[[ATOM_M,ATOM_K], BM/ATOM_M, BK/ATOM_K]`; for TMA the atom is usually the whole tile → `[[...],1,1]` → one instruction.
- **TMA multicast slicing** = slicing the *inverse* of the smem layout: split along slowest-moving dim first so each CTA's chunk is contiguous (`tma_partition`).
- **PyTorch ↔ CuTe**: `randn(M,N,K)` → `(M,N,K):(NK,K,1)`; transpose/permute swaps shape+stride together (no copy); `.contiguous()` copies into LayoutRight.
- Swizzle layouts (composition with a `Swizzle<B,M,S>` functor) are the MMA/TMA smem layouts — see [[swizzling]]; TV layouts (thread-value) are covered in the author's "4 Ways to Do CuTe Copy" post (not digested here).

## Numbers worth remembering
- Example: M0=2,M1=2,N0=2,N1=3 blocked product; coord `((0,1),(1,1))` → 2D `(2,3)` → index 18.

## Code patterns
```cpp
using namespace cute;
// PyTorch W[n,k] contiguous (row-major) as a CuTe gmem tensor
auto gW = make_tensor(make_gmem_ptr(W), make_layout(make_shape(N, K), make_stride(K, _1{})));
// CTA tile of W: rows [bn*BN, (bn+1)*BN), cols [kt*BK, ...)
auto tW = local_tile(gW, Tile<Int<BN>, Int<BK>>{}, make_coord(bn, kt));
// whole K-loop view: keep k-tile index as a mode (underscore = "all")
auto tWk = local_tile(gW, Tile<Int<BN>, Int<BK>>{}, make_coord(bn, _));   // [BN, BK, K/BK]
// static smem layout (required by make_tma_copy)
auto sL = make_layout(make_shape(Int<BN>{}, Int<BK>{}), make_stride(Int<BK>{}, _1{}));
auto L1 = coalesce(make_layout(make_shape(4, 8), make_stride(1, 4)));     // -> 32:1
print(layout(tW));                                                        // debug: always print layouts
```

## Pitfalls
- Forgetting CuTe is column-major by default (wrong strides silently produce transposed math).
- Dynamic ints in smem layouts → `make_tma_copy` compile errors; dynamic tile shapes defeat compile-time checks of TMA/MMA invariants.
- Confusing `size` (number of elements) with `shape` (tuple) and `cosize` (span in memory; matters for smem allocation with padded/swizzled layouts).

## Relevance to the serving track
- `out[m,n] = x[m,k] · W[n,k]ᵀ`: both x and W are "K-major" (k contiguous) — the natural TN GEMM layout; no transpose kernel needed, describe W as `(n,k):(k,1)`. (inference)
- Bucketed m ∈ {1,8,16,32,128,256}: make BM a static template param per bucket (CuTe/CUTLASS kernels are templated on static tile shapes anyway). (inference)
- For m=1 GEMV-style kernels you may skip CuTe entirely; for wgmma/TMA kernels CuTe layouts are the least error-prone way to get swizzled smem + descriptors right. (inference)

## Related
[[yangyifan-cute-layouts]] [[tma]] [[swizzling]] [[wgmma]] [[mma-ptx]] [[register-tiling]]
