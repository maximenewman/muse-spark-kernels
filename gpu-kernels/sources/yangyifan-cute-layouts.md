---
title: "Yifan Yang – CuTe Layout and Tensor"
type: source
tags: [gpu, cute, cutlass, layout, tiling]
source_url: https://yang-yifan.github.io/blogs/cute_layout/cute_layout.html
code_url: https://github.com/Yang-YiFan/Yang-YiFan.github.io/tree/main/blogs/cute_layout/code
---
# Yifan Yang – CuTe Layout and Tensor

Fetched in full 2026-11-08. An intuition-first companion to the official CuTe docs (01_layout, 02_layout_algebra, 03_tensor). Does **not** cover composition, TV layouts, swizzle (deferred to future posts; TV layouts appear in "4 Ways to Do CuTe Copy").

**TL;DR**
- Layout = `Shape : Stride`, a function from coordinates to a physical index. Modes = dimensions, rank = number of modes.
- Every layout is equivalently natural-coord → index, or 1D-coord → index (1D coord = colexicographic/LayoutLeft linearization of the shape). Size = |domain|, cosize = |codomain|.
- `coalesce` flattens modes when `stride[i+1] == shape[i]*stride[i]` (or size-1 modes); `coalesce(L, Step<_1,_1,_1>{})` does it per mode, preserving rank.
- `blocked_product(tile, tiler)` repeats a tile; `local_tile(T, tiler, coord)` = numpy slice `T[m1*M0:(m1+1)*M0, n1*N0:(n1+1)*N0]`.
- Prefer **static** ints; rule of thumb: RMEM/SMEM/TMEM layouts are always static. TMA/MMA can only take certain tile shapes, so static shapes let the compiler check them.

## Key ideas
- Default stride is **LayoutLeft** (column-major: stride[0]=1, prefix products left→right). LayoutRight = row-major. `(M,N,K):(1,M,MN)` vs `(M,N,K):(NK,K,1)`.
- Hierarchical shapes: `((2,2),2):((4,1),2)` is still a 4×2 matrix; natural coords mirror the shape nesting `((m,n),k)`.
- Coalesce rules (from CuTe docs): `(s0,_1):(d0,d1) → s0:d0`; `(_1,s1):(d0,d1) → s1:d1`; `(s0,s1):(d0,s0*d0) → s0*s1:d0`.
- **Blocked product**: tile `(M0,N0):(1,M0)`, row-major tiler `(M1,N1):(N1,1)` → `((M0,M1),(N0,N1)) : ((1, N1*M0*N0), (M0, M0*N0))`; N mode further coalesces to `((M0,M1), N0*N1) : ((1, N1*M0*N0), M0)`. Built from two 1D **logical products**, but not independently — M1's stride needs N0,N1, so blocked product does M then N with the dependency.
- **local_tile** = `inner_partition`; tiler is a tuple of shapes (not a layout) because modes are partitioned independently; it need not invert the blocked_product that built the tensor.
- **PyTorch mapping**: `torch.randn(M,N,K)` → `(M,N,K):(NK,K,1)`; `.T`/`permute` permutes shape *and* stride together (storage unchanged): `A (M,N):(N,1)` → `A.T (N,M):(1,N)`; `.contiguous()` materializes a LayoutRight copy.

## Numbers worth remembering
- Worked example: M0=2, M1=2, N0=2, N1=3, coord `((0,1),(1,1))` = 2D `(2,3)` → storage[18].

## Code patterns
```cpp
auto L  = make_layout(make_shape(4,2), make_stride(2,1));              // 4x2 row-major: A(m,n)=m*2+n
auto L2 = coalesce(L3, Step<_1,_1,_1>{});                                // by-mode coalesce, keep rank 3
auto B  = blocked_product(tile, tiler);                                  // repeat tile per tiler order
auto t  = local_tile(gA, Tile<Int<BM>,Int<BK>>{}, make_coord(bm, bk));  // CTA's sub-tile view
auto S  = make_layout(make_shape(Int<BN>{},Int<BK>{}), make_stride(Int<BK>{},_1{}));  // static smem layout
```

## Pitfalls
- CuTe's default is column-major — opposite of PyTorch/C. Write strides explicitly for row-major `W[n,k]`.
- Transposing in PyTorch (`W.T`) does not move data; a CuTe layout of the same storage must swap strides, not copy.
- Dynamic ints in smem layouts break `make_tma_copy` compilation (see [[yangyifan-cute-tma]]).

## Relevance to the serving track
- `x @ W.T` with PyTorch `W` of shape [n,k] contiguous = CuTe `(n,k):(k,1)`; viewed as the B operand `(k,n)` it is `(k,n):(1,k)` — "K-major" B, the layout wgmma/TMA prefer without transposition. (inference)
- Static tile shapes for the m∈{1,8,16,32,128,256} buckets → one template instantiation per bucket so TMA boxes / MMA atoms are compile-time checked. (inference)

## Further reading
See [[yangyifan-cute-tma]] "Further reading". Author's references: CuTe docs 01/02/03, Cris Cecka's GPU MODE lecture, Reed's Zhihu CuTe series.

## Related
[[cute-layouts]] [[tma]] [[swizzling]] [[wgmma]] [[register-tiling]]
