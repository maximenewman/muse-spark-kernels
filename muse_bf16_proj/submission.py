# okc: problem=muse_bf16_proj
"""Skinny BF16 projections (and the FP32 router) as split-K Triton kernels.

Every scored shape is weight-bandwidth bound, so the kernels are built to stream W at
HBM speed: the weight is the "A" operand (swap-AB, W[BN, BK] @ x^T[BK, BM]) so the
token count only sets the narrow side of the MMA, and K is split across CTAs so that
even the small layers fill all SMs. Split-K partials go to an FP32 workspace and the
last CTA to finish a tile (atomic ticket) reduces them and writes the output, then
resets its ticket, so one launch does the whole GEMM and graph replays need no memset.

m == 1 uses a CUDA-core GEMV (FP32 weights read directly for the router).
The router at m >= 2 uses the BF16 hi + lo split of its FP32 weight, FP32 accumulation.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl

import skinny_cuda


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# ----------------------------------------------------------------------------- kernels


@triton.jit
def _splitk_epilogue(acc, OUT, WS, CNT, tile_id, pid_k, offs_r, offs_c, mask, stride_r, stride_c,
                     R: tl.constexpr, C: tl.constexpr, SPLIT_K, OUT_F32: tl.constexpr):
    """acc is an [R, C] FP32 tile of the output at rows offs_r / cols offs_c (strided).
    SPLIT_K == 1 stores it; otherwise partials meet in WS and the last CTA reduces."""
    out_ptrs = OUT + offs_r[:, None] * stride_r + offs_c[None, :] * stride_c
    if SPLIT_K == 1:
        if OUT_F32:
            tl.store(out_ptrs, acc, mask=mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        loc = tl.arange(0, R)[:, None] * C + tl.arange(0, C)[None, :]
        ws_tile = WS + tile_id.to(tl.int64) * (SPLIT_K * R * C)
        tl.store(ws_tile + pid_k * (R * C) + loc, acc, cache_modifier=".cg")
        tl.debug_barrier()
        ticket = tl.atomic_add(CNT + tile_id, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT_K - 1:
            total = tl.zeros((R, C), dtype=tl.float32)
            # 4 independent partial loads in flight per step (a serial loop is one L2 round-trip per split)
            for s0 in range(0, SPLIT_K, 4):
                for u in tl.static_range(4):
                    total += tl.load(ws_tile + (s0 + u) * (R * C) + loc, mask=(loc >= 0) & (s0 + u < SPLIT_K),
                                     other=0.0, cache_modifier=".cg")
            if OUT_F32:
                tl.store(out_ptrs, total, mask=mask)
            else:
                tl.store(out_ptrs, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(CNT + tile_id, 0, sem="relaxed", scope="gpu")


@triton.jit
def _gemm_kernel(X, W, W2, OUT, WS, CNT, M, stride_xm, stride_wn, stride_om,
                 K_SPLIT, SPLIT_K, BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                 HILO: tl.constexpr, OUT_F32: tl.constexpr):
    """out[m, n] = sum_k x[m, k] W[n, k] (+ W2[n, k]), as W[BN, BK] @ x^T[BK, BM]."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    k0 = pid_k * K_SPLIT
    offs_k = k0 + tl.arange(0, BK)
    w_ptrs = W + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    w2_ptrs = W2 + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = X + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for _ in range(0, K_SPLIT, BK):
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        xt = tl.load(x_ptrs, mask=m_mask[None, :], other=0.0)
        acc = tl.dot(w, xt, acc)
        if HILO:
            w2 = tl.load(w2_ptrs, eviction_policy="evict_first")
            acc = tl.dot(w2, xt, acc)
            w2_ptrs += BK
        w_ptrs += BK
        x_ptrs += BK
    tile_id = pid_m * tl.num_programs(0) + pid_n
    # out is [M, N]: tile rows are n (stride 1), cols are m (stride stride_om)
    _splitk_epilogue(acc, OUT, WS, CNT, tile_id, pid_k, offs_n, offs_m, m_mask[None, :], 1, stride_om,
                     BN, BM, SPLIT_K, OUT_F32)


@triton.jit
def _mm_kernel(X, W, W2, OUT, WS, CNT, M, stride_xm, stride_wn, stride_om, K_SPLIT, SPLIT_K,
               BN: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, HILO: tl.constexpr,
               OUT_F32: tl.constexpr):
    """Standard orientation for larger m: x[BM, BK] @ W^T[BK, BN] (wgmma on Hopper)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    m_mask = offs_m < M
    k0 = pid_k * K_SPLIT
    offs_k = k0 + tl.arange(0, BK)
    x_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = W + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    w2_ptrs = W2 + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K_SPLIT, BK):
        xa = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        wt = tl.load(w_ptrs, eviction_policy="evict_first")
        acc = tl.dot(xa, wt, acc)
        if HILO:
            wt2 = tl.load(w2_ptrs, eviction_policy="evict_first")
            acc = tl.dot(xa, wt2, acc)
            w2_ptrs += BK
        x_ptrs += BK
        w_ptrs += BK
    tile_id = pid_m * tl.num_programs(0) + pid_n
    _splitk_epilogue(acc, OUT, WS, CNT, tile_id, pid_k, offs_m, offs_n, m_mask[:, None], stride_om, 1,
                     BM, BN, SPLIT_K, OUT_F32)


@triton.jit
def _gemv_kernel(X, W, OUT, WS, CNT, stride_wn,
                 K_SPLIT, SPLIT_K, BN: tl.constexpr, BK: tl.constexpr, OUT_F32: tl.constexpr):
    """m == 1: out[n] = sum_k x[k] W[n, k] on CUDA cores, FP32 accumulation."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    k0 = pid_k * K_SPLIT
    offs_k = k0 + tl.arange(0, BK)
    w_ptrs = W + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = X + offs_k
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for _ in range(0, K_SPLIT, BK):
        w = tl.load(w_ptrs, eviction_policy="evict_first")
        x = tl.load(x_ptrs)
        acc += w.to(tl.float32) * x.to(tl.float32)[None, :]
        w_ptrs += BK
        x_ptrs += BK
    r = tl.sum(acc, axis=1)[:, None]  # [BN, 1]
    zero = tl.arange(0, 1)
    _splitk_epilogue(r, OUT, WS, CNT, pid_n, pid_k, offs_n, zero, offs_n[:, None] >= 0, 1, 0,
                     BN, 1, SPLIT_K, OUT_F32)


# ----------------------------------------------------------------------------- configs

# (kind, BN, BM, BK, SPLIT_K, num_warps, num_stages); kind "gemv" ignores BM.
# Tuned per scored shape on H200 with graph replays and a cold L2 (tune.py).
CONFIGS: dict = {
    ('o', 1): ('gemv', 16, 1, 512, 1, 4, 1),
    ('pooler', 1): ('gemv', 8, 1, 512, 1, 8, 1),
    ('qkv_gate', 1): ('gemv', 4, 1, 1024, 1, 4, 1),
    ('router', 1): ('gemv', 4, 1, 1024, 4, 4, 3),
    ('unpooler', 1): ('gemv', 4, 1, 1024, 1, 8, 3),
    ('o', 8): ('gemm', 64, 16, 128, 1, 4, 5),
    ('pooler', 8): ('gemm', 32, 16, 256, 1, 4, 5),
    ('qkv_gate', 8): ('gemm', 64, 16, 128, 2, 4, 5),
    ('router', 8): ('cuda', 1, 2, 4, 16),  # hand-written CUDA (skinny_cuda.py), 5.44 vs 5.79 µs
    ('unpooler', 8): ('gemm', 32, 16, 128, 4, 4, 5),
    ('o', 16): ('gemm', 64, 16, 128, 1, 4, 5),
    ('pooler', 16): ('gemm', 32, 16, 256, 1, 4, 5),
    ('qkv_gate', 16): ('gemm', 64, 16, 128, 2, 4, 5),
    ('router', 16): ('gemm', 32, 16, 128, 16, 4, 5),
    ('unpooler', 16): ('gemm', 64, 16, 128, 1, 4, 5),
    ('o', 32): ('gemm', 64, 32, 128, 1, 4, 5),
    ('pooler', 32): ('gemm', 32, 32, 256, 1, 4, 5),
    ('qkv_gate', 32): ('gemm', 64, 32, 128, 2, 4, 5),
    ('router', 32): ('gemm', 16, 32, 128, 8, 4, 5),
    ('unpooler', 32): ('gemm', 64, 32, 128, 1, 4, 5),
    ('router', 128): ('mm', 64, 64, 128, 8, 4, 4),
    ('router', 256): ('mm', 64, 128, 128, 8, 8, 3),
}
# Above this many tokens the BF16 layers go to cuBLAS (our swap-AB tiles lose there for now).
CUBLAS_MIN_M = 33


def _default_config(layer: str, m: int, n: int, k: int):
    sms = _NUM_SMS
    if m == 1:
        bn, bk = 16, 512
        tiles = n // bn
        split = 1
        while tiles * split < 4 * sms and k // (split * 2) >= bk * 2:
            split *= 2
        return ("gemv", bn, 1, bk, split, 4, 3)
    bm = 16 if m <= 16 else 32 if m <= 32 else 64 if m <= 64 else 128
    if m <= 64:
        bn, bk = 64, 128
    else:
        bn, bk = 128, 64
    tiles = (n // bn) * triton.cdiv(m, bm)
    split = 1
    while tiles * split < 2 * sms and k // (split * 2) >= bk * 2:
        split *= 2
    warps = 4 if bn * bm <= 64 * 32 else 8
    return ("gemm", bn, bm, bk, split, warps, 4)


def pick_config(layer: str, m: int, n: int, k: int):
    if (layer, m) in CONFIGS:
        return CONFIGS[(layer, m)]
    # nearby m: the next tuned bucket up (same unpooler width split at 8 tokens)
    for bucket in (1, 8, 16, 32, 128, 256):
        if m <= bucket and (bucket > 1 or m == 1):
            cfg = CONFIGS.get((layer, bucket))
            if cfg is not None and (cfg[0] == "gemv") == (m == 1):
                return cfg
            break
    return _default_config(layer, m, n, k)


# ----------------------------------------------------------------------------- host side

_NUM_SMS = 132
WS_BYTES = 64 << 20
N_COUNTERS = 1 << 16


def prepare(weights: Weights):
    global _NUM_SMS
    _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    dev = weights["router"].device
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"].contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["ws"] = torch.empty(WS_BYTES // 4, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(N_COUNTERS, dtype=torch.int32, device=dev)
    skinny_cuda.module()  # compile the CUDA extension (untimed)
    return state


def run(x: torch.Tensor, layer: str, n: int, state, cfg=None) -> torch.Tensor:
    m, k = x.shape
    out_f32 = layer == "router"
    if m >= CUBLAS_MIN_M and not out_f32 and cfg is None and (layer, m) not in CONFIGS:
        return torch.mm(x, state[layer][:n].t())
    if m > 256 or (out_f32 and m > 128 and cfg is None and (layer, m) not in CONFIGS):  # cuBLAS
        if out_f32:
            out = torch.mm(x, state["router_hi"].t(), out_dtype=torch.float32)
            out += torch.mm(x, state["router_lo"].t(), out_dtype=torch.float32)
            return out
        return torch.mm(x, state[layer][:n].t())
    out = torch.empty((m, n), dtype=torch.float32 if out_f32 else torch.bfloat16, device=x.device)
    cfg = cfg or pick_config(layer, m, n, k)
    if cfg[0] == "ring":
        _, bk, stages, warps, split = cfg[:5]
        assert skinny_cuda.ring_valid(m, n, k, bk, stages, warps, split, out_f32)
        assert (n // (16 * warps)) * split * 16 * warps * 8 * ((m + 7) // 8) * 4 <= WS_BYTES
        return skinny_cuda.run_ring(x, state[layer], out, state["ws"], state["cnt"], bk, stages, warps, split)
    if cfg[0] == "cuda":
        _, r, u, warps, split = cfg[:5]
        w = state["router"] if out_f32 else state[layer]
        assert skinny_cuda.valid(m, n, k, r, u, warps, split, out_f32)
        assert (n // (16 * r)) * split * 16 * r * 8 * ((m + 7) // 8) * 4 <= WS_BYTES
        return skinny_cuda.run(x, w, out, state["ws"], state["cnt"], r, u, warps, split)
    kind, bn, bm, bk, split, warps, stages = cfg
    k_split = k // split
    assert n % bn == 0 and k_split % bk == 0 and k % split == 0
    ws, cnt = state["ws"], state["cnt"]
    if kind == "gemv":
        w = state["router"] if out_f32 else state[layer]
        assert (n // bn) * split * bn * 4 <= WS_BYTES
        _gemv_kernel[(n // bn, split)](x, w, out, ws, cnt, w.stride(0),
                                       k_split, split, BN=bn, BK=bk, OUT_F32=out_f32,
                                       num_warps=warps, num_stages=stages)
        return out
    if out_f32:
        w, w2, hilo = state["router_hi"], state["router_lo"], True
    else:
        w, w2, hilo = state[layer], state[layer], False
    grid_m = triton.cdiv(m, bm)
    if kind == "mm":
        assert (n // bn) * grid_m * split * bn * bm * 4 <= WS_BYTES or split == 1
        _mm_kernel[(n // bn, split, grid_m)](x, w, w2, out, ws, cnt, m, x.stride(0), w.stride(0), out.stride(0),
                                             k_split, split, BN=bn, BM=bm, BK=bk, HILO=hilo,
                                             OUT_F32=out_f32, num_warps=warps, num_stages=stages)
        return out
    assert (n // bn) * grid_m * split * bn * bm * 4 <= WS_BYTES and (n // bn) * grid_m <= N_COUNTERS
    _gemm_kernel[(n // bn, split, grid_m)](x, w, w2, out, ws, cnt, m, x.stride(0), w.stride(0), out.stride(0),
                                           k_split, split, BN=bn, BM=bm, BK=bk, HILO=hilo,
                                           OUT_F32=out_f32, num_warps=warps, num_stages=stages)
    return out


def kernel_fn(data: Inputs, state) -> torch.Tensor:
    cfg = data["config"]
    return run(data["x"], cfg["layer"], cfg["n"], state)
