---
title: WGMMA (Hopper warpgroup MMA)
type: concept
tags: [gpu, hopper, sm90a, wgmma, tensor-cores, ptx, gemm]
source_url:
  - https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
  - https://github.com/pranjalssh/fast.cu  (h100/matmul/matmul_8.cuh)
  - https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell
  - https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/
---
# WGMMA (`wgmma.mma_async`, Hopper sm_90a)

**TL;DR**
- Hopper's tensor-core instruction, issued **asynchronously by a warpgroup** (4 contiguous warps, 128 threads). SASS name `HGMMA`/`GMMA`.
- Shape `m64nNk16` for 16-bit inputs: **M fixed at 64**, N = 8…256 in steps of 8, K = 16 (K = 32 for FP8).
- A from SMEM (64-bit descriptor) or registers; **B always from SMEM**; FP32 accumulator in **registers** (64·N/128 regs/thread).
- You must bracket it: `wgmma.fence` → N× `wgmma.mma_async` → `wgmma.commit_group` → `wgmma.wait_group n`.
- Only in `sm_90a` (arch-specific). Gone on Blackwell, which uses [[tcgen05-tmem]].
- For skinny GEMMs (m ≤ 256 tokens), map **weights to M=64 and tokens to N** (inference).

## Key ideas
- **Async + warpgroup scope.** The 128 threads jointly issue one MMA whose accumulator fragment is spread over all of them. Execution overlaps with subsequent instructions until `wait_group`. [semianalysis, worklog]
- **Operand sources.** A: SMEM descriptor or RF fragment. B: SMEM descriptor only. D: RF. [semianalysis] SMEM tiles must be in a canonical (usually 128B-swizzled, K-major) layout, which [[tma]] produces directly. [worklog]
- **Larger N is faster.** It amortizes A re-reads from SMEM. The worklog went 64×64 → `m64n128k16` (317 → 423 TFLOPs) and finally `m64n256k16`. [worklog]
- **Register cost.** A 64×N FP32 accumulator = 64·N/128 = N/2 regs/thread. n256 → 128 regs/thread; two m64 tiles per warpgroup (BM=128 per consumer) → 256, which spills. [worklog]
- **Scale-d.** The `scale-d` immediate/predicate selects D = A·B (0) vs D = A·B + D (1). Using 0 on the first k-step avoids zeroing the accumulators (worklog kernel 9). Other immediates: scale-a/scale-b (±1) and trans-a/trans-b (transpose bits for 16-bit types).

## Numbers worth remembering
- Shapes (bf16/fp16 → f32): `m64n{8,16,...,256}k16`. [worklog, semianalysis]
- SMEM per instruction: m64n16k16 A+B = 2.5 KB. n256 ≈ 40 KB of SMEM operands for the tile the worklog used. [worklog]
- Accumulator fragment layout (from fast.cu store code): thread `tid` in the warpgroup, `warp = tid/32`, `lane = tid%32`, `row = warp*16 + lane/4`, for each 16-column chunk `w`, `col = w + 2*(tid%4)`. Its 8 floats `d[w/16][0..7]` are (row,col), (row,col+1), (row+8,col), (row+8,col+1), (row,col+8), (row,col+9), (row+8,col+8), (row+8,col+9).
- H100: 4 tensor cores/SM, 132 SMs. [worklog]

## Code patterns
Fence / commit / wait (fast.cu):
```cuda
asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");          // order prior RF/SMEM writes before wgmma reads
asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");   // close a group of issued wgmmas
asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory"); // wait until <= N groups pending (0..7)
```
One instruction, m64n64k16, both operands from SMEM descriptors (fast.cu `wgmma64`, operand list abbreviated):
```cuda
asm volatile("{\n wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
  "{%0, ..., %31},"          // 32 f32 accumulators per thread (64*64/128)
  " %32,"                    // desc_a (uint64 smem descriptor)
  " %33,"                    // desc_b
  " %34, %35, %36, %37, %38;\n}\n"   // scale-d, scale-a, scale-b, trans-a, trans-b (immediates)
  : "+f"(d[0][0]), ... : "l"(desc_a), "l"(desc_b),
    "n"(ScaleD), "n"(ScaleA), "n"(ScaleB), "n"(TransA), "n"(TransB));
```
Descriptor for 128B-swizzled bf16 tile: start>>4 in bits 0–13, LBO (16 B → 1) at bit 16, SBO (1024 B = 8 rows × 128 B → 64) at bit 32, bit 62 = 128B swizzle. See [[h100-gemm-worklog-cudaforfun]] for the full function and [[swizzling]] for the layout. Advancing along K inside a 128B swizzle atom: the worklog just offsets the base pointer by `k_it*16` elements (32 B) per k16 step.

Consumer k-loop:
```cuda
wait(&full[q], phase);                     // TMA data landed (see warp-specialization)
warpgroup_arrive();                         // wgmma.fence
for (int k_it = 0; k_it < BK/16; ++k_it)
    wgmma<BN,1,1,1,0,0>(d, &sA[k_it*16], &sB[k_it*16]);
warpgroup_commit_batch();
warpgroup_wait<0>();
arrive(&empty[q]);                          // release smem slot to producer
```

## Pitfalls
- **Spills serialize wgmma.** ptxas warns C7511 "wgmma.mma_async instructions are serialized". The worklog fell from ~500 to 123 TFLOPs. Keep accumulators ≤ ~128–200 regs/thread, split M across consumer warpgroups, or use `setmaxnreg` ([[warp-specialization]]).
- **Accumulator registers must not be touched** between issue and `wait_group` (inference from async semantics; the fence/wait discipline above enforces it).
- Must compile for `sm_90a` (not `sm_90`). The worklog builds with `arch=compute_90a,code=sm_90a`.
- wgmma isn't in the CUDA C++ programming guide. The worklog author says the PTX docs' swizzle diagrams are wrong; let TMA produce the swizzle.
- Releasing an SMEM slot (`arrive(empty)`) before the wgmma reading it has completed is a race. Wait on the group first.

## Relevance to the serving track
- (inference) **Swap AB for small m.** out[m,n] = x[m,k]·W[n,k]ᵀ. Compute outᵀ[n,m] = W·xᵀ with W as wgmma A (M=64 rows of W per instruction, K-major since W is [n,k] row-major) and xᵀ as B (N = m padded to a multiple of 8; x is [m,k] row-major = K-major for B). m ∈ {1,8,16,32,128,256} → N ∈ {8,8,16,32,128,256}. Accumulator cost is only N/2 regs/thread (4 regs for m≤8), so registers are never the constraint here.
- (inference) At m ≤ 32 the tensor core is >95% idle anyway. wgmma is still worth using over SIMT FMA because it issues with tiny instruction count and lets one warpgroup consume a 64×K weight tile straight from swizzled SMEM. A well-written `mma.sync` m16n8k16 or even FMA GEMV can match it when HBM-bound ([[mma-ptx]]). Benchmark both.
- (inference) Bytes per SM matter, not FLOPs. A 64-row × BK=64 bf16 W tile is only 8 KB, so use deep stages (e.g. 6–10 × 16–32 KB) to keep ≥ ~40–60 KB in flight per SM and cover HBM latency.
- (inference) The router layer in FP32 via BF16 hi+lo uses 2–3 wgmmas per k-step on split operands. Compute is free here, but split in registers/SMEM after loading FP32 once. Don't store W twice.
- FP8 wgmma (k32) exists if a problem allows FP8 weights (halves bytes). Accumulation precision caveat in [[semianalysis-tensor-core-evolution]].

## Related
[[tensor-core-generations]] · [[warp-specialization]] · [[tma]] · [[swizzling]] · [[smem-bank-conflicts]] · [[mma-ptx]] · [[register-tiling]] · [[tcgen05-tmem]] · [[h100-gemm-worklog-cudaforfun]] · [[semianalysis-tensor-core-evolution]]
