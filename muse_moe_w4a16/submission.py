# okc: problem=muse_moe_w4a16
"""Routed experts, NVFP4 weights, BF16 activations: route + two weight-streaming GEMMs + norm-sum.

Kernels (raw CUDA, compiled in-process with NVRTC via torch.cuda._compile_kernel):
  route   one CTA: groups the local (token, slot) pairs by expert, writes work items
          (expert, first sorted position, tokens) and zeroes the norm accumulators.
  gemm1   per (row tile, item): h = x @ W13^T with W13 dequantized in registers (FP4 ->
          BF16, block scale applied with one bf16x2 multiply), mma.sync m16n8k16 with the
          weights as A (16 rows: 8 gate + the matching 8 up) and tokens as B; SiLU(gate)*up
          in the epilogue -> inter [sorted pos, I] bf16.
  gemm2   y = inter @ W2^T, epilogue y*(1+w) -> bf16 and the row's partial sum of squares
          (atomicAdd per sorted position).
  final   per token: sum over its local slots of weight * yw * rsqrt(ss / H + eps).

The GEMMs stream weights through a shared-memory ring filled by one producer warp with
cp.async.bulk (1-D TMA) + mbarriers; prepare() lays every CTA's per-stage chunk out
contiguously, in mma-fragment order (one u32 = one thread's 8 FP4 values for one k-step).
"""

import ctypes
from typing import TypedDict

import torch

from nvfp4 import dequantize


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8]
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


H = 4096
I = 4096
TOPK = 8
NLOC = 32
WP = 4           # warps per packed weight tile (prepare); a CTA runs W in {2, 4, 8} consumer warps
KSS = 16         # k-steps (16 K each) per pipeline stage
NSTG = H // (16 * KSS)  # stages per K sweep (K = 4096 for both GEMMs)
EPS = 1e-8

COMMON = r"""
typedef unsigned int u32;
typedef unsigned short u16;

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 a, u32 c) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(c)); }
__device__ __forceinline__ void mbar_expect_tx(u32 a, u32 b) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(b) : "memory"); }
__device__ __forceinline__ void mbar_arrive(u32 a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void mbar_wait(u32 a, u32 parity) {
  asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n" :: "r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_g2s(u32 dst, const void* src, u32 bytes, u32 mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(mbar) : "memory");
}
__device__ __forceinline__ u32 bmul2(u32 a, u32 b) { u32 d; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// prepare() places each FP4 code's 3 magnitude bits and sign so that a shift + mask lands them at
// bf16 bits 6-8 and 15 (and +16): two bf16 equal to e2m1 * 2^-126 (subnormal for the 0.5 code).
#define FP4M 0x81C081C0u
__device__ __forceinline__ void unp4(u32 q, u32& a0, u32& a1, u32& a2, u32& a3) {
  a0 = q & FP4M;
  a1 = (q << 3) & FP4M;
  a2 = (q << 6) & FP4M;
  a3 = ((q >> 4) & 0x00C000C0u) | ((q >> 5) & 0x01000100u) | ((q << 1) & 0x80008000u);
}
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(u32& r0, u32& r1, u32& r2, u32& r3, u32 addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {  // round to nearest even
  u32 u = __float_as_uint(f); u += 0x7FFFu + ((u >> 16) & 1u); return (u16)(u >> 16);
}
"""

ROUTE_SRC = COMMON + r"""
extern "C" __global__ void moe_route(const int* __restrict__ ids, int P, int lstart, int TB,
    int* __restrict__ sorted_tok, int* __restrict__ pos_of, int* __restrict__ items, int* __restrict__ n_items,
    float* __restrict__ ss) {
  __shared__ int cnt[32], off[32], cur[32], ioff[32];
  int tid = threadIdx.x;
  if (tid < 32) { cnt[tid] = 0; cur[tid] = 0; }
  __syncthreads();
  for (int i = tid; i < P; i += blockDim.x) {
    int e = ids[i] - lstart;
    if (e >= 0 && e < 32) atomicAdd(&cnt[e], 1);
    ss[i] = 0.f;
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0, si = 0;
    for (int e = 0; e < 32; ++e) { off[e] = s; s += cnt[e]; ioff[e] = si; si += (cnt[e] + TB - 1) / TB; }
    *n_items = si;
  }
  __syncthreads();
  for (int i = tid; i < P; i += blockDim.x) {
    int e = ids[i] - lstart;
    if (e >= 0 && e < 32) { int p = off[e] + atomicAdd(&cur[e], 1); sorted_tok[p] = i >> 3; pos_of[i] = p; }
    else pos_of[i] = -1;
  }
  if (tid < 32) {
    int c = cnt[tid];
    for (int b = 0; b * TB < c; ++b) {
      int it = ioff[tid] + b;
      items[3 * it] = tid; items[3 * it + 1] = off[tid] + b * TB; items[3 * it + 2] = min(TB, c - b * TB);
    }
  }
}
"""

FINAL_SRC = COMMON + r"""
extern "C" __global__ void moe_final(const u16* __restrict__ yw, const float* __restrict__ ss, const int* __restrict__ pos_of,
    const float* __restrict__ tw, u16* __restrict__ out) {
  const float eps = 1e-8f;
  int t = blockIdx.x;
  int c = threadIdx.x * 8;
  float acc[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = 0.f;
  #pragma unroll
  for (int s = 0; s < 8; ++s) {
    int p = pos_of[t * 8 + s];
    if (p >= 0) {
      float r = rsqrtf(ss[p] * (1.0f / 4096.0f) + eps) * tw[t * 8 + s];
      uint4 v = *(const uint4*)(yw + (size_t)p * 4096 + c);
      u32 w4[4] = {v.x, v.y, v.z, v.w};
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[2 * j] += r * __uint_as_float(w4[j] << 16);
        acc[2 * j + 1] += r * __uint_as_float(w4[j] & 0xFFFF0000u);
      }
    }
  }
  uint4 o;
  o.x = f2bf(acc[0]) | ((u32)f2bf(acc[1]) << 16);
  o.y = f2bf(acc[2]) | ((u32)f2bf(acc[3]) << 16);
  o.z = f2bf(acc[4]) | ((u32)f2bf(acc[5]) << 16);
  o.w = f2bf(acc[6]) | ((u32)f2bf(acc[7]) << 16);
  *(uint4*)(out + (size_t)t * 4096 + c) = o;
}
"""

# MODE 1: gate/up + SiLU (A rows: 8 gate + 8 up per warp tile, 8 outputs / warp)
# MODE 2: down + (1+w) + partial sum of squares (16 outputs / warp)
GEMM_SRC = COMMON + r"""
#define A_BYTES (W * KSS * 128)
#define SF_BYTES (W * 16 * KSS)
#define XROW (KSS * 16 * 2)
#define XSTRIDE (XROW + 16)
#define X_BYTES (TB * XSTRIDE)
#define STAGE_BYTES (A_BYTES + SF_BYTES + X_BYTES)
#define NB (TB / 8)
#if MODE == 1
#define CT (4096 / (8 * W))
#define CTP (4096 / (8 * WP))
#else
#define CT (4096 / (16 * W))
#define CTP (4096 / (16 * WP))
#endif
#define CW (W < WP ? W : WP)      // warps per bulk copy (contiguous within a packed tile)
#define NCH (W / CW)              // copies per stage for A and for SF

extern "C" __global__ void __launch_bounds__((W + 1) * 32) moe_gemm(
    const u32* __restrict__ wq, const unsigned char* __restrict__ wsf, const float* __restrict__ gsc,
    const u16* __restrict__ xsrc, const int* __restrict__ sorted_tok, const int* __restrict__ items,
    const int* __restrict__ n_items, u16* __restrict__ out, const u16* __restrict__ normw, float* __restrict__ ss,
    float* __restrict__ ws, int* __restrict__ cnt) {
  // split-K: blockIdx.z of gridDim.z takes stages [k0, k1); partial sums go to ws (fp32, zero between
  // uses) and the last CTA of each (item, row tile) runs the epilogue and re-zeroes ws / cnt.
  extern __shared__ __align__(128) unsigned char smem[];
  __shared__ int s_last;
  const int KS = gridDim.z;
  const int k0 = blockIdx.z * NSTG / KS, k1 = (blockIdx.z + 1) * NSTG / KS;
  const int nI = *n_items;
#if ITEM_FAST
  const int slot0 = blockIdx.x, nslot = gridDim.x, ct = blockIdx.y;
#else
  const int slot0 = blockIdx.y, nslot = gridDim.y, ct = blockIdx.x;
#endif
  if (slot0 >= nI) return;
  unsigned long long* bars = (unsigned long long*)smem;   // full[NST], empty[NST]
  unsigned char* stages = smem + 128;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NST; ++s) { mbar_init(smem_u32(&bars[s]), 1); mbar_init(smem_u32(&bars[NST + s]), W); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == W) {  // producer
    if (lane == 0) {
      int it = 0;
      for (int item = slot0; item < nI; item += nslot) {
        const int e = items[3 * item], ps = items[3 * item + 1], n = items[3 * item + 2];
        const u32* wsrc[NCH];
        const unsigned char* ssrc[NCH];
        #pragma unroll
        for (int c = 0; c < NCH; ++c) {
          const int gw = ct * W + c * CW, pt = gw / WP, wl = gw % WP;
          wsrc[c] = wq + ((size_t)(e * CTP + pt) * NSTG) * (WP * KSS * 32) + wl * KSS * 32;
          ssrc[c] = wsf + ((size_t)(e * CTP + pt) * NSTG) * (WP * 16 * KSS) + wl * 16 * KSS;
        }
        int rows[TB];
        #pragma unroll
        for (int r = 0; r < TB; ++r) {
#if MODE == 1
          rows[r] = r < n ? sorted_tok[ps + r] : 0;
#else
          rows[r] = ps + r;
#endif
        }
        for (int k = k0; k < k1; ++k, ++it) {
          const int st = it % NST, rnd = it / NST;
          if (rnd > 0) mbar_wait(smem_u32(&bars[NST + st]), (rnd - 1) & 1);
          const u32 fb = smem_u32(&bars[st]);
          unsigned char* sp = stages + st * STAGE_BYTES;
          mbar_expect_tx(fb, A_BYTES + SF_BYTES + n * XROW);
          #pragma unroll
          for (int c = 0; c < NCH; ++c) {
            bulk_g2s(smem_u32(sp + c * CW * KSS * 128), wsrc[c] + (size_t)k * (WP * KSS * 32), CW * KSS * 128, fb);
            bulk_g2s(smem_u32(sp + A_BYTES + c * CW * 16 * KSS), ssrc[c] + (size_t)k * (WP * 16 * KSS), CW * 16 * KSS, fb);
          }
          #pragma unroll
          for (int r = 0; r < TB; ++r)
            if (r < n) bulk_g2s(smem_u32(sp + A_BYTES + SF_BYTES + r * XSTRIDE), xsrc + (size_t)rows[r] * 4096 + k * KSS * 16, XROW, fb);
        }
      }
    }
    return;
  }

  // consumers
  const int g = lane >> 2, t = lane & 3;
  int it = 0;
  for (int item = slot0; item < nI; item += nslot) {
    const int e = items[3 * item], ps = items[3 * item + 1], n = items[3 * item + 2];
    float acc[NB][4];
    #pragma unroll
    for (int b = 0; b < NB; ++b) { acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f; }
    for (int k = k0; k < k1; ++k, ++it) {
      const int st = it % NST, rnd = it / NST;
      mbar_wait(smem_u32(&bars[st]), rnd & 1);
      const unsigned char* sp = stages + st * STAGE_BYTES;
#ifdef NOCOMPUTE
      if (true) { __syncwarp(); if (lane == 0) mbar_arrive(smem_u32(&bars[NST + st])); continue; }
#endif
      const uint4* As = (const uint4*)sp + warp * (KSS / 4) * 32 + lane;
      const unsigned char* Ss = sp + A_BYTES + warp * 16 * KSS;
      const u32 xs = smem_u32(sp + A_BYTES + SF_BYTES);
      // ldmatrix row address for this lane: token (lane & 7), k offset (lane >> 3) * 8
      const u32 xl = xs + (lane & 7) * XSTRIDE + ((lane >> 4) * 16 + ((lane >> 3) & 1) * 8) * 2;
      #pragma unroll
      for (int kq = 0; kq < KSS / 4; ++kq) {
        const uint4 aw = As[kq * 32];
        const u32 sg = *(const u32*)(Ss + g * KSS + kq * 4);
        const u32 sg8 = *(const u32*)(Ss + (g + 8) * KSS + kq * 4);
        u32 bq[NB][8];
        #pragma unroll
        for (int b = 0; b < NB; ++b) {
          if (b > 0 && b * 8 >= n) break;  // b = 0 always has tokens
          ldsm4(bq[b][0], bq[b][1], bq[b][2], bq[b][3], xl + b * 8 * XSTRIDE + (kq * 4) * 32);
          ldsm4(bq[b][4], bq[b][5], bq[b][6], bq[b][7], xl + b * 8 * XSTRIDE + (kq * 4 + 2) * 32);
        }
        const u32 q4[4] = {aw.x, aw.y, aw.z, aw.w};
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          const u32 sel = 0x4040u + kk * 0x101u;
          const u32 S0 = prmt(sg, 0, sel) * 16u + 0x77807780u;
          const u32 S1 = prmt(sg8, 0, sel) * 16u + 0x77807780u;
          const u32 q = q4[kk];
          u32 a0, a1, a2, a3;
          unp4(q, a0, a1, a2, a3);
          a0 = bmul2(a0, S0); a1 = bmul2(a1, S1); a2 = bmul2(a2, S0); a3 = bmul2(a3, S1);
          #pragma unroll
          for (int b = 0; b < NB; ++b) { if (b > 0 && b * 8 >= n) break; mma16816(acc[b], a0, a1, a2, a3, bq[b][2 * kk], bq[b][2 * kk + 1]); }
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&bars[NST + st]));
    }
#if MODE == 1
#define WSW 8192
    const int ra = ct * (8 * W) + warp * 8 + g, rb = 4096 + ra;   // gate row, up row
#else
#define WSW 4096
    const int ra = ct * (16 * W) + warp * 16 + g, rb = ra + 8;
#endif
    if (KS > 1) {
      #pragma unroll
      for (int b = 0; b < NB; ++b)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tok = b * 8 + 2 * t + h;
          if (tok < n) {
            atomicAdd(&ws[(size_t)(ps + tok) * WSW + ra], acc[b][h]);
            atomicAdd(&ws[(size_t)(ps + tok) * WSW + rb], acc[b][2 + h]);
          }
        }
      __threadfence();
      asm volatile("bar.sync 1, %0;" :: "r"(W * 32) : "memory");
      if (threadIdx.x == 0) {
        const int tk = atomicAdd(&cnt[item * CT + ct], 1);
        s_last = (tk == KS - 1);
        if (s_last) cnt[item * CT + ct] = 0;
      }
      asm volatile("bar.sync 1, %0;" :: "r"(W * 32) : "memory");
      if (!s_last) continue;
      __threadfence();
      #pragma unroll
      for (int b = 0; b < NB; ++b)
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tok = b * 8 + 2 * t + h;
          if (tok < n) {
            float* pa = &ws[(size_t)(ps + tok) * WSW + ra];
            float* pb = &ws[(size_t)(ps + tok) * WSW + rb];
            acc[b][h] = __ldcg(pa); acc[b][2 + h] = __ldcg(pb);
            *pa = 0.f; *pb = 0.f;
          }
        }
    }
    const float gs = gsc[e];
#if MODE == 1
    const int col = ct * (8 * W) + warp * 8 + g;
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tok = b * 8 + 2 * t + h;
        if (tok < n) {
          const float gt = acc[b][h] * gs, up = acc[b][2 + h] * gs;
          const float v = gt / (1.f + __expf(-gt)) * up;
          out[(size_t)(ps + tok) * 4096 + col] = f2bf(v);
        }
      }
    }
#else
    const int r0 = ct * (16 * W) + warp * 16 + g;
    const float w0 = 1.f + bf2f(normw[r0]), w1 = 1.f + bf2f(normw[r0 + 8]);
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
      #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tok = b * 8 + 2 * t + h;
        const u16 y0 = f2bf(acc[b][h] * gs * w0), y1 = f2bf(acc[b][2 + h] * gs * w1);
        float sq = bf2f(y0) * bf2f(y0) + bf2f(y1) * bf2f(y1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (tok < n) {
          out[(size_t)(ps + tok) * 4096 + r0] = y0;
          out[(size_t)(ps + tok) * 4096 + r0 + 8] = y1;
          if (g == 0) atomicAdd(&ss[ps + tok], sq);
        }
      }
    }
#endif
  }
}
"""

# ---------------------------------------------------------------- NVRTC + driver, in-process
# Own ctypes wrapper: libraries are loaded once by soname (no find_library, no subprocesses),
# kernels compile to a cubin with NVRTC and launch with cuLaunchKernel on torch's stream.
_LIB = {}


def _libs():
    if not _LIB:
        major = int(torch.version.cuda.split(".")[0])
        for name in (f"libnvrtc.so.{major}", "libnvrtc.so"):
            try:
                _LIB["nvrtc"] = ctypes.CDLL(name)
                break
            except OSError:
                continue
        _LIB["cuda"] = ctypes.CDLL("libcuda.so.1")
        _LIB["cuda"].cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
    return _LIB["nvrtc"], _LIB["cuda"]


def _ck(r, what):
    if r != 0:
        raise RuntimeError(f"{what} failed with error {r}")


class _Kernel:
    def __init__(self, src, name):
        nvrtc, cuda = _libs()
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # make torch's primary context current
        prog = ctypes.c_void_p()
        _ck(nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), f"{name}.cu".encode(), 0, None, None),
            "nvrtcCreateProgram")
        cc = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{cc[0]}{cc[1]}{'a' if cc[0] == 9 else ''}".encode(), b"-std=c++17",
                b"--use_fast_math", b"-default-device"]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        r = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)
        if r != 0:
            n = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value)
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC compile of {name} failed:\n{log.value.decode()}")
        n = ctypes.c_size_t()
        _ck(nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(n)), "nvrtcGetCUBINSize")
        self.cubin = ctypes.create_string_buffer(n.value)
        _ck(nvrtc.nvrtcGetCUBIN(prog, self.cubin), "nvrtcGetCUBIN")
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        self.mod = ctypes.c_void_p()
        _ck(cuda.cuModuleLoadData(ctypes.byref(self.mod), self.cubin), "cuModuleLoadData")
        self.func = ctypes.c_void_p()
        _ck(cuda.cuModuleGetFunction(ctypes.byref(self.func), self.mod, name.encode()), "cuModuleGetFunction")
        self.smem = 0

    def set_smem(self, nbytes):
        if nbytes > self.smem:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
            _ck(_libs()[1].cuFuncSetAttribute(self.func, 8, int(nbytes)), "cuFuncSetAttribute")
            self.smem = nbytes

    def __call__(self, grid, block, args, shared_mem=0):
        keep = []
        for a in args:
            if isinstance(a, torch.Tensor):
                keep.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, float):
                keep.append(ctypes.c_float(a))
            else:
                keep.append(ctypes.c_int(a))
        ptrs = (ctypes.c_void_p * len(keep))(*[ctypes.addressof(k) for k in keep])
        stream = torch.cuda.current_stream().cuda_stream
        _ck(_libs()[1].cuLaunchKernel(self.func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                      shared_mem, stream, ptrs, None), "cuLaunchKernel")


_K = {}


def _compile(src, name, defines=None):
    head = "".join(f"#define {k} {v}\n" for k, v in (defines or {}).items())
    key = (name, head)
    if key not in _K:
        _K[key] = _Kernel(head + src, name)
    return _K[key]


ITEM_FAST = 0  # 1: grid x = item slot (items of one expert together); measured no better


EXTRA_DEFS = {}  # experiments only (e.g. NOCOMPUTE)


def _gemm(mode, tb, nst, w):
    k = _compile(GEMM_SRC, "moe_gemm", dict(MODE=mode, TB=tb, W=w, WP=WP, KSS=KSS, NSTG=NSTG, NST=nst,
                                            ITEM_FAST=ITEM_FAST, **EXTRA_DEFS))
    k.set_smem(_smem(tb, nst, w))
    return k


def _smem(tb, nst, w=WP):
    stage = w * KSS * 128 + w * 16 * KSS + tb * (KSS * 32 + 16)
    return 128 + nst * stage


# ---------------------------------------------------------------- weight packing (untimed)

def _pack(w, sf, rows_map):
    """w uint8 [E, R, K/2], sf e4m3 [E, R, K/16], rows_map long [NWT, 16] (logical rows of each
    warp tile) -> (u32 words [E, CT, NSTG, W, KSS/4, 32, 4], scale bytes [E, CT, NSTG, W, 16, KSS])."""
    E, R, Kh = w.shape
    K = Kh * 2
    NS = K // 16
    NWT = rows_map.shape[0]
    CT = NWT // WP
    dev = w.device
    g = torch.arange(8, device=dev)
    t = torch.arange(4, device=dev)
    words, scales = [], []
    for e in range(E):
        codes = torch.stack((w[e] & 15, w[e] >> 4), -1).reshape(R, K)
        A = codes[rows_map].view(NWT, 16, NS, 16).to(torch.int64)  # [wt, row, s, kk]
        word = torch.zeros((NWT, NS, 8, 4), dtype=torch.int64, device=dev)
        for j in range(4):
            r = (g + 8 * (j % 2))[:, None]          # [8, 1]
            kk = (2 * t + 8 * (j // 2))[None, :]     # [1, 4]
            lo = A[:, r, :, kk]                      # [8, 4, NWT, NS]
            hi = A[:, r, :, kk + 1]
            lo = lo.permute(2, 3, 0, 1)
            hi = hi.permute(2, 3, 0, 1)
            for h, c in ((0, lo), (1, hi)):
                mag, sg = c & 7, c >> 3
                if j < 3:
                    word |= (mag << (6 - 3 * j + 16 * h)) | (sg << (15 - 3 * j + 16 * h))
                else:
                    word |= ((mag & 3) << (10 + 16 * h)) | ((mag >> 2) << (13 + 16 * h)) | (sg << (14 + 16 * h))
        word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
        # [NWT, NS, g, t] -> [CT, W, NSTG, KSS/4, 4, 32] -> [CT, NSTG, W, KSS/4, 32, 4]
        word = word.view(CT, WP, NSTG, KSS // 4, 4, 32).permute(0, 2, 1, 3, 5, 4).contiguous()
        words.append(word)
        s = sf[e][rows_map].view(torch.uint8)  # [NWT, 16, NS]
        s = (s & 0x7F).view(CT, WP, 16, NSTG, KSS).permute(0, 3, 1, 2, 4).contiguous()
        scales.append(s)
    return torch.stack(words), torch.stack(scales)


def _scales_ok(sf):
    b = sf.view(torch.uint8)
    e = (b >> 3) & 0xF
    return bool(((b & 0x80) == 0).all() and (e != 0).all() and ((b & 0x7F) != 0x7F).all())


def prepare(weights):
    dev = weights["w13"].device
    sl = slice(0, NLOC)
    wt = torch.arange(I // 8, device=dev)
    r = torch.arange(8, device=dev)
    rows1 = torch.cat((8 * wt[:, None] + r[None, :], I + 8 * wt[:, None] + r[None, :]), 1)  # gate 8, up 8
    rows2 = 16 * torch.arange(H // 16, device=dev)[:, None] + torch.arange(16, device=dev)[None, :]
    fast = _scales_ok(weights["w13_sf"][sl]) and _scales_ok(weights["w2_sf"][sl])
    st = {"fast": fast, "weights": weights}
    if not fast:
        return st
    st["w1q"], st["w1s"] = _pack(weights["w13"][sl], weights["w13_sf"][sl], rows1)
    st["w2q"], st["w2s"] = _pack(weights["w2"][sl], weights["w2_sf"][sl], rows2)
    st["g1"] = (weights["w13_gs"][sl].float() * 128.0).contiguous()
    st["g2"] = (weights["w2_gs"][sl].float() * 128.0).contiguous()
    st["normw"] = weights["post_norm_w"].contiguous()
    # split-K workspace (fp32 partial sums, kept zero between uses) and per-(item, row tile) tickets
    st["ws"] = torch.zeros((WS_PAIRS, 2 * I), dtype=torch.float32, device=dev)
    st["cnt"] = torch.zeros(WS_ITEMS * (I // (8 * 2)), dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    return st


# ---------------------------------------------------------------- timed call

# M -> (tokens per item, pipeline stages, grid slots); tuned on H100 (tune.py)
WS_PAIRS = 512      # split-K only for M * 8 <= WS_PAIRS
WS_ITEMS = 256
CFG = {1: (8, 4, 4), 16: (8, 3, 4), 32: (8, 2, 16), 8: (8, 2, 4), 128: (16, 2, 16), 256: (16, 2, 64)}
CFG_OVERRIDE = {}


def _cfg(M):
    """(tokens per item, stages, grid slots, split-K gemm1, split-K gemm2, warps per CTA)"""
    c = tuple(CFG_OVERRIDE.get(M) or CFG.get(M) or ((8 if M <= 64 else 16), 3, 16))
    c = c + (1, 1, WP)[len(c) - 3:]
    if M * TOPK > WS_PAIRS:
        c = c[:3] + (1, 1) + c[5:]
    return c


def kernel_fn(data, state):
    if not state["fast"]:
        return _slow(data, state["weights"])
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].contiguous()
    tw = data["topk_weights"].contiguous()
    M = x.shape[0]
    P = M * TOPK
    dev = x.device
    tb, nst, nslot, ks1, ks2, W = _cfg(M)
    maxi = min(NLOC, P) + (P + tb - 1) // tb
    sorted_tok = torch.empty(P, dtype=torch.int32, device=dev)
    pos_of = torch.empty(P, dtype=torch.int32, device=dev)
    items = torch.empty(3 * maxi, dtype=torch.int32, device=dev)
    n_items = torch.empty(1, dtype=torch.int32, device=dev)
    ss = torch.empty(P, dtype=torch.float32, device=dev)
    inter = torch.empty((P, I), dtype=torch.bfloat16, device=dev)
    yw = torch.empty((P, H), dtype=torch.bfloat16, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    _compile(ROUTE_SRC, "moe_route")((1, 1, 1), (1024, 1, 1),
                                     (ids, P, int(cfg["local_start"]), tb, sorted_tok, pos_of, items, n_items, ss))
    slots = min(maxi, nslot)
    smem = _smem(tb, nst, W)
    g1 = _gemm(1, tb, nst, W)
    grid1 = (slots, I // (8 * W), ks1) if ITEM_FAST else (I // (8 * W), slots, ks1)
    grid2 = (slots, H // (16 * W), ks2) if ITEM_FAST else (H // (16 * W), slots, ks2)
    g1(grid1, ((W + 1) * 32, 1, 1),
       (state["w1q"], state["w1s"], state["g1"], x, sorted_tok, items, n_items, inter, state["normw"], ss, state["ws"], state["cnt"]),
       shared_mem=smem)
    g2 = _gemm(2, tb, nst, W)
    g2(grid2, ((W + 1) * 32, 1, 1),
       (state["w2q"], state["w2s"], state["g2"], inter, sorted_tok, items, n_items, yw, state["normw"], ss, state["ws"], state["cnt"]),
       shared_mem=smem)
    _compile(FINAL_SRC, "moe_final")((M, 1, 1), (512, 1, 1), (yw, ss, pos_of, tw, out))
    return out


def _slow(data, weights):
    cfg = data["config"]
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    w1 = (weights["post_norm_w"].float() + 1.0)
    for e in range(cfg["local_start"], cfg["local_start"] + cfg["local_count"]):
        g = (data["topk_weights"] * (ids == e)).sum(-1)
        h = x @ dequantize(weights["w13"][e], weights["w13_sf"][e], weights["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        y = (inter @ dequantize(weights["w2"][e], weights["w2_sf"][e], weights["w2_gs"][e]).T) * w1
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
        out += g[:, None] * y
    return out.to(torch.bfloat16)
