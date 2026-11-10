"""Hand-written CUDA skinny GEMM for m <= 32: out[m, n] = x[m, k] @ W[n, k]^T.

Design (gpu-kernels vault, gemv-and-skinny-gemm option iii):
- W is streamed straight from HBM into registers with 16 B ld.global.nc.L1::no_allocate loads,
  in mma.sync A-fragment order. No shared memory, no ldmatrix. Thread (g, t) of a warp
  owns W rows g and g+8 of a 16-row fragment and the 16 consecutive k at [16t, 16t+16) of
  each 64-k block. W and x use the same k-permutation, so the dot product is unchanged:
      mma step s (0..3), logical k 2t+j    <-> actual k 16t + 4s + j
                         logical k 2t+8+j  <-> actual k 16t + 4s + 2 + j
  Lanes 4g..4g+3 read 128 contiguous bytes of row g: full sectors.
- Swap-AB mma.sync m16n8k16 (W rows -> M = 16, tokens -> N = 8), f32 accumulation. At m = 1
  the 7 unused token columns are free: the kernel is HBM bound.
- FP32 router weights: loaded as f32, split in registers into BF16 hi + lo (hi = rn(w),
  lo = rn(w - hi)), two mma per step into the same f32 accumulator.
- A CTA owns 16*R rows. Its warps split the CTA's K slice and reduce through shared memory;
  CTAs split K (SPLIT) and the last one to finish a row block (atomic ticket, self-reset)
  sums the partials and writes the output. One kernel, no memset, graph-replay safe.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// k-permutation layouts (element offsets within a 64-k block). WV = uint4 per row per thread.
#ifdef SKINNY_V2
// v2: each thread owns 16 contiguous k (32 B bf16 / 64 B f32)
#define W_OFF(v, t) (16 * (t) + (v) * (16 / WV))
#define XB_OFF(i, t) (16 * (t) + 8 * (i))
#define XF_OFF(s, t) (16 * (t) + 4 * (s))
#else
// v3: for each load instruction the 4 lanes of a row read 64 contiguous bytes
#define W_OFF(v, t) ((v) * (64 / WV) + (t) * (16 / WV))
#define XB_OFF(i, t) (32 * (i) + 8 * (t))
#define XF_OFF(s, t) (16 * (s) + 4 * (t))
#endif

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint4 ldg_x(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo_k, float hi_k) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo_k, hi_k);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Split two consecutive f32 weights into (hi pair, lo pair) of packed bf16.
__device__ __forceinline__ void split2(uint32_t w0, uint32_t w1, uint32_t& hi, uint32_t& lo) {
  float f0 = __uint_as_float(w0), f1 = __uint_as_float(w1);
  __nv_bfloat162 h = __floats2bfloat162_rn(f0, f1);
  float2 hf = __bfloat1622float2(h);
  hi = *reinterpret_cast<uint32_t*>(&h);
  lo = pack_bf16(f0 - hf.x, f1 - hf.y);
}

__device__ __forceinline__ uint32_t u4get(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int NT, int R, int U, bool F32W>
struct Batch {
  static constexpr int WEL = F32W ? 4 : 2;
  static constexpr int WV = WEL;
  uint4 w[U][R][2][WV];

  __device__ __forceinline__ void load(const char* Wb, int row0, int g, int t, int K, int kb) {
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const size_t row = row0 + r * 16 + g + h * 8;
          // uint4 v of this thread covers k [v*64/WV + t*16/WV, +16/WV): for each load
          // instruction the 4 lanes of a row read 64 contiguous bytes (full sectors).
          const char* p = Wb + ((size_t)row * K + kb + 64 * u) * WEL;
#pragma unroll
          for (int v = 0; v < WV; ++v) w[u][r][h][v] = ldg_stream(p + W_OFF(v, t) * WEL);
        }
  }

  __device__ __forceinline__ void compute(float (&acc)[R][NT][4], const __nv_bfloat16* __restrict__ X,
                                          int M, int K, int kb, int g, int t) const {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      // xb[j][s] = the B fragment (2 x b32) of step s, using W's k-permutation:
      //   bf16 W: step s <-> k 32*(s/2) + 8t + 4*(s%2) + 0..3
      //   f32 W:  step s <-> k 16*s + 4t + 0..3
      uint32_t xb[NT][4][2];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int tok = j * 8 + g;
        const __nv_bfloat16* p = X + (size_t)tok * K + kb + 64 * u;
        if (F32W) {
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            uint2 v = make_uint2(0, 0);
            if (tok < M) v = *reinterpret_cast<const uint2*>(p + XF_OFF(s, t));
            xb[j][s][0] = v.x; xb[j][s][1] = v.y;
          }
        } else {
          uint4 x0 = make_uint4(0, 0, 0, 0), x1 = make_uint4(0, 0, 0, 0);
          if (tok < M) { x0 = ldg_x(p + XB_OFF(0, t)); x1 = ldg_x(p + XB_OFF(1, t)); }
          xb[j][0][0] = x0.x; xb[j][0][1] = x0.y; xb[j][1][0] = x0.z; xb[j][1][1] = x0.w;
          xb[j][2][0] = x1.x; xb[j][2][1] = x1.y; xb[j][3][0] = x1.z; xb[j][3][1] = x1.w;
        }
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        // registers of this thread's 16 k used by step s: elements 4s..4s+3
#pragma unroll
        for (int r = 0; r < R; ++r) {
          uint32_t a[2][2], al[2][2];  // [h][pair]: pair 0 = k 4s,4s+1 ; pair 1 = k 4s+2,4s+3
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (F32W) {
              // 16 f32 = 4 uint4; element e lives in w[..][e/4] component e%4
              const int e = 4 * s;
              split2(u4get(w[u][r][h][e / 4], 0), u4get(w[u][r][h][e / 4], 1), a[h][0], al[h][0]);
              split2(u4get(w[u][r][h][e / 4], 2), u4get(w[u][r][h][e / 4], 3), a[h][1], al[h][1]);
            } else {
              // 16 bf16 = 2 uint4; pair index p = 2s + q lives in w[..][p/4] component p%4
              a[h][0] = u4get(w[u][r][h][(2 * s) / 4], (2 * s) % 4);
              a[h][1] = u4get(w[u][r][h][(2 * s + 1) / 4], (2 * s + 1) % 4);
            }
          }
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const uint32_t b0 = xb[j][s][0], b1 = xb[j][s][1];
            mma16816(acc[r][j], a[0][0], a[1][0], a[0][1], a[1][1], b0, b1);
            if (F32W) mma16816(acc[r][j], al[0][0], al[1][0], al[0][1], al[1][1], b0, b1);
          }
        }
      }
    }
  }
};

// NT: token tiles of 8 (m <= 8*NT). R: 16-row fragments per warp. U: 64-k blocks per load batch.
template <int NT, int R, int U, bool F32W, bool OUT_F32>
__global__ void __launch_bounds__(256) skinny_kernel(
    const __nv_bfloat16* __restrict__ X, const void* __restrict__ Wv, void* __restrict__ OUT,
    float* __restrict__ WS, int* __restrict__ CNT, int M, int N, int K, int k_per_warp, int split) {
  constexpr int WEL = F32W ? 4 : 2;            // bytes per weight element
  constexpr int WV = WEL;                       // uint4 per (row, 64-k block) per thread: 16 k * WEL B / 16 B
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * (16 * R);
  const int k0 = blockIdx.y * (k_per_warp * nwarps) + warp * k_per_warp;
  const char* Wb = reinterpret_cast<const char*>(Wv);

  float acc[R][NT][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][j][i] = 0.f;

  using B = Batch<NT, R, U, F32W>;
  // Double-buffered register pipeline: batch i+1's W loads are in flight while batch i is computed.
  B ba, bb;
  const int kend = k0 + k_per_warp;
  constexpr int STEP = 64 * U;
  ba.load(Wb, row0, g, t, K, k0);
  for (int kb = k0; kb < kend; kb += 2 * STEP) {
    if (kb + STEP < kend) bb.load(Wb, row0, g, t, K, kb + STEP);
    ba.compute(acc, X, M, K, kb, g, t);
    if (kb + STEP >= kend) break;
    if (kb + 2 * STEP < kend) ba.load(Wb, row0, g, t, K, kb + 2 * STEP);
    bb.compute(acc, X, M, K, kb + STEP, g, t);
  }

  // ---- reduce across warps: red[warp][row (16R)][token (8NT)]
  constexpr int ROWS = 16 * R, TOK = 8 * NT, TILE = ROWS * TOK;
  extern __shared__ float red[];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = r * 16 + g + (i >= 2 ? 8 : 0);
        const int tok = j * 8 + 2 * t + (i & 1);
        red[warp * TILE + tok * ROWS + row] = acc[r][j][i];
      }
  __syncthreads();
  for (int e = threadIdx.x; e < TILE; e += blockDim.x) {
    float sum = 0.f;
    for (int w2 = 0; w2 < nwarps; ++w2) sum += red[w2 * TILE + e];
    red[e] = sum;  // only warp 0's slot is overwritten, and only by the thread that read it
  }
  __syncthreads();

  auto store = [&](int e, float v) {
    const int tok = e / ROWS, row = e % ROWS;
    if (tok < M) {
      const size_t o = (size_t)tok * N + row0 + row;
      if (OUT_F32) reinterpret_cast<float*>(OUT)[o] = v;
      else reinterpret_cast<__nv_bfloat16*>(OUT)[o] = __float2bfloat16_rn(v);
    }
  };
  if (split == 1) {
    for (int e = threadIdx.x; e < TILE; e += blockDim.x) store(e, red[e]);
    return;
  }
  float* ws = WS + (size_t)blockIdx.x * split * TILE;
  for (int e = threadIdx.x; e < TILE; e += blockDim.x) __stcg(ws + blockIdx.y * TILE + e, red[e]);
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = (atomicAdd(CNT + blockIdx.x, 1) == split - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int e = threadIdx.x; e < TILE; e += blockDim.x) {
    float sum = 0.f;
#pragma unroll 8
    for (int s2 = 0; s2 < split; ++s2) sum += __ldcg(ws + s2 * TILE + e);
    store(e, sum);
  }
  if (threadIdx.x == 0) CNT[blockIdx.x] = 0;
}

template <int NT, int R, int U, bool F32W, bool OUT_F32>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, torch::Tensor& ws,
            torch::Tensor& cnt, int warps, int split) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  const int k_per_warp = K / (split * warps);
  dim3 grid(N / (16 * R), split);
  const int smem = warps * 16 * R * 8 * NT * 4;
  auto kern = skinny_kernel<NT, R, U, F32W, OUT_F32>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kern<<<grid, warps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), w.data_ptr(), out.data_ptr(),
      ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K, k_per_warp, split);
}

#define CASE_U(NT, R, F32W, OUT_F32)                                              \
  if (u == 1) return launch<NT, R, 1, F32W, OUT_F32>(x, w, out, ws, cnt, warps, split); \
  if (u == 2) return launch<NT, R, 2, F32W, OUT_F32>(x, w, out, ws, cnt, warps, split); \
  if (!F32W && u == 4) return launch<NT, R, 4, F32W, OUT_F32>(x, w, out, ws, cnt, warps, split);
#define CASE_R(NT, F32W, OUT_F32) \
  if (r == 1) { CASE_U(NT, 1, F32W, OUT_F32) } \
  if (r == 2) { CASE_U(NT, 2, F32W, OUT_F32) }

void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
            int64_t r, int64_t u, int64_t warps, int64_t split) {
  const int nt = (x.size(0) + 7) / 8;
  const bool f32w = w.scalar_type() == torch::kFloat32;
  if (f32w) {
    if (nt == 1) { CASE_R(1, true, true) }
    if (nt == 2) { CASE_R(2, true, true) }
    if (nt <= 4) { CASE_R(4, true, true) }
  } else {
    if (nt == 1) { CASE_R(1, false, false) }
    if (nt == 2) { CASE_R(2, false, false) }
    if (nt <= 4) { CASE_R(4, false, false) }
  }
  TORCH_CHECK(false, "skinny: unsupported config");
}

// ---------------------------------------------------------------------------------------------
// v4: W streamed through a STAGES-deep cp.async ring in shared memory (bytes in flight live in
// smem, not registers), 16 B chunks XOR-swizzled by (row & 7) so the fragment LDS.128s are
// conflict-free. Each warp owns 16 rows (BN = 16 * warps), so no cross-warp reduction.
// Fragment mapping = the v2 k-permutation (thread t owns k [16t, 16t+16) of each 64-k block).
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N_> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N_)); }
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 r;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a));
  return r;
}

template <int NT, int BK, int STAGES>
__global__ void __launch_bounds__(256) ring_kernel(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ OUT,
    float* __restrict__ WS, int* __restrict__ CNT, int M, int N, int K, int k_per_cta, int split) {
  extern __shared__ __align__(128) unsigned char smem[];
  constexpr int CPR = BK / 8;  // 16 B chunks per row
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int BN = 16 * nwarps;
  const int row0 = blockIdx.x * BN;
  const int kbeg = blockIdx.y * k_per_cta;
  const int iters = k_per_cta / BK;
  constexpr int TOK = 8 * NT;
  const int w_bytes = BN * BK * 2;
  const int stage_bytes = (BN + TOK) * BK * 2;  // W tile, then the x tile [TOK][BK] (zero rows >= M)
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));

  auto issue = [&](int it) {
    if (it < iters) {
      const int kk = kbeg + it * BK;
      const uint32_t st = sbase + (it % STAGES) * stage_bytes;
      for (int c = threadIdx.x; c < BN * CPR; c += blockDim.x) {
        const int row = c / CPR, ch = c % CPR;
        cp_async16(st + (row * CPR + (ch ^ (row & 7))) * 16, W + (size_t)(row0 + row) * K + kk + ch * 8);
      }
      for (int c = threadIdx.x; c < TOK * CPR; c += blockDim.x) {
        const int tok = c / CPR, ch = c % CPR;
        const bool live = tok < M;
        cp_async16_zfill(st + w_bytes + (tok * CPR + (ch ^ (tok & 7))) * 16,
                         X + (size_t)(live ? tok : 0) * K + kk + ch * 8, live ? 16 : 0);
      }
    }
    cp_async_commit();
  };

  float acc[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) issue(s);
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  for (int it = 0; it < iters; ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    issue(it + STAGES - 1);
    const uint32_t st = sbase + (it % STAGES) * stage_bytes;
#pragma unroll
    for (int jb = 0; jb < BK / 64; ++jb) {
      const int c0 = jb * 8 + 2 * t, c1 = c0 + 1;
      uint4 w[2][2];
      w[0][0] = lds128(st + (r0 * CPR + (c0 ^ (r0 & 7))) * 16);
      w[0][1] = lds128(st + (r0 * CPR + (c1 ^ (r0 & 7))) * 16);
      w[1][0] = lds128(st + (r1 * CPR + (c0 ^ (r1 & 7))) * 16);
      w[1][1] = lds128(st + (r1 * CPR + (c1 ^ (r1 & 7))) * 16);
      uint4 xv[NT][2];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int tok = j * 8 + g;
        xv[j][0] = lds128(st + w_bytes + (tok * CPR + (c0 ^ (tok & 7))) * 16);
        xv[j][1] = lds128(st + w_bytes + (tok * CPR + (c1 ^ (tok & 7))) * 16);
      }
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const uint32_t a0 = u4get(w[0][s / 2], 2 * (s % 2)), a2 = u4get(w[0][s / 2], 2 * (s % 2) + 1);
        const uint32_t a1 = u4get(w[1][s / 2], 2 * (s % 2)), a3 = u4get(w[1][s / 2], 2 * (s % 2) + 1);
#pragma unroll
        for (int j = 0; j < NT; ++j)
          mma16816(acc[j], a0, a1, a2, a3, u4get(xv[j][s / 2], 2 * (s % 2)), u4get(xv[j][s / 2], 2 * (s % 2) + 1));
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  // ---- epilogue: tile [token (8NT)][row (BN)] in smem, then store or split-K ticket
  const int TILE = BN * TOK;
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i)
      red[(j * 8 + 2 * t + (i & 1)) * BN + warp * 16 + g + (i >= 2 ? 8 : 0)] = acc[j][i];
  __syncthreads();
  auto store = [&](int e, float v) {
    const int tok = e / BN, row = e % BN;
    if (tok < M) OUT[(size_t)tok * N + row0 + row] = __float2bfloat16_rn(v);
  };
  if (split == 1) {
    for (int e = threadIdx.x; e < TILE; e += blockDim.x) store(e, red[e]);
    return;
  }
  float* ws = WS + (size_t)blockIdx.x * split * TILE;
  for (int e = threadIdx.x; e < TILE; e += blockDim.x) __stcg(ws + blockIdx.y * TILE + e, red[e]);
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = (atomicAdd(CNT + blockIdx.x, 1) == split - 1);
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int e = threadIdx.x; e < TILE; e += blockDim.x) {
    float sum = 0.f;
#pragma unroll 8
    for (int s2 = 0; s2 < split; ++s2) sum += __ldcg(ws + s2 * TILE + e);
    store(e, sum);
  }
  if (threadIdx.x == 0) CNT[blockIdx.x] = 0;
}

template <int NT, int BK, int STAGES>
void launch_ring(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& out, torch::Tensor& ws,
                 torch::Tensor& cnt, int warps, int split) {
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  const int BN = 16 * warps;
  const int smem = std::max(STAGES * (BN + 8 * NT) * BK * 2, BN * 8 * NT * 4);
  auto kern = ring_kernel<NT, BK, STAGES>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kern<<<dim3(N / BN, split), warps * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), ws.data_ptr<float>(), cnt.data_ptr<int>(), M, N, K,
      K / split, split);
}

#define RING_ST(NT, BK) \
  if (stages == 3) return launch_ring<NT, BK, 3>(x, w, out, ws, cnt, warps, split); \
  if (stages == 4) return launch_ring<NT, BK, 4>(x, w, out, ws, cnt, warps, split); \
  if (stages == 6) return launch_ring<NT, BK, 6>(x, w, out, ws, cnt, warps, split);
#define RING_BK(NT) \
  if (bk == 64) { RING_ST(NT, 64) } \
  if (bk == 128) { RING_ST(NT, 128) }

void ring(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
          int64_t bk, int64_t stages, int64_t warps, int64_t split) {
  const int nt = (x.size(0) + 7) / 8;
  if (nt == 1) { RING_BK(1) }
  if (nt == 2) { RING_BK(2) }
  if (nt <= 4) { RING_BK(4) }
  TORCH_CHECK(false, "ring: unsupported config");
}
"""

CPP_SRC = ("void skinny(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t r, int64_t u, int64_t warps, int64_t split);\n"
           "void ring(torch::Tensor x, torch::Tensor w, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt, int64_t bk, int64_t stages, int64_t warps, int64_t split);")

_mods = {}


def module(layout: str = "v3"):
    """layout "v3" (default): full-sector loads per instruction; "v2": 32 contiguous B per thread."""
    if layout not in _mods:
        flags = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"]
        if layout == "v2":
            flags.append("-DSKINNY_V2")
        _mods[layout] = load_inline(name=f"muse_skinny_{layout}_r7", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                                    functions=["skinny", "ring"], extra_cuda_cflags=flags, verbose=False)
    return _mods[layout]


def valid(m, n, k, r, u, warps, split, f32w):
    if m > 32 or n % (16 * r) or (f32w and u > 2):
        return False
    return k % (split * warps * 64 * u) == 0


def run(x, w, out, ws, cnt, r, u, warps, split, layout="v3"):
    module(layout).skinny(x, w, out, ws, cnt, r, u, warps, split)
    return out


def ring_valid(m, n, k, bk, stages, warps, split, f32w):
    bn = 16 * warps
    return (not f32w and m <= 32 and n % bn == 0 and k % (split * bk) == 0 and k // split >= bk
            and stages * (bn + 8 * ((m + 7) // 8)) * bk * 2 <= 200 * 1024)


def run_ring(x, w, out, ws, cnt, bk, stages, warps, split):
    module().ring(x, w, out, ws, cnt, bk, stages, warps, split)
    return out
