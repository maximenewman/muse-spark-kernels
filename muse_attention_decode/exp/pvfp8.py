# okc: problem=muse_attention_decode
"""Gated attention decode over a paged FP8 cache: two Triton kernels.

1. _prep_kernel (one program per new token): RMSNorm of q (x qk_scale_factor) and k, interleaved
   RoPE on sliding layers, q stored as bf16, k and v cast to FP8 (divided by the cache scales)
   and written to the token's cache row.
2. _attn_kernel (grid batch x kv_head x split): flash-decoding over this split's slice of the
   context. The group's query heads (x the block's L tokens) form the MMA's M rows; keys are
   gathered row by row through the page table (FP8 -> bf16), causal and window masks per row.
   Each split writes its (m, l, acc) to a workspace; the last split of a (batch, kv head) pair
   (atomic ticket, self-resetting) merges them and applies rmsnorm(a) * sigmoid(gate), so the
   whole attention + combine + norm-gate is one launch and graph replays need no memset.
"""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634


@triton.jit
def _prep_kernel(QKV, POS, LOC, QOUT, KC, VC, INV_FREQ, KSC, VSC, stride_qkv,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
                 QK_SCALE: tl.constexpr, EPS: tl.constexpr, QFP8: tl.constexpr):
    """One program per new token; all heads at once (one load per tensor, no per-head chain).
    QFP8: q stored as E4M3 of the normalized (RoPE'd) query, as FA3's FP8 path rounds it; the
    qk factor then moves into the softmax scale."""
    tok = tl.program_id(0)
    i = tl.arange(0, D // 2)
    base = QKV + tok * stride_qkv
    pos = tl.load(POS + tok).to(tl.float32)
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    qe = tl.load(base + hq[:, None] * D + 2 * i[None, :]).to(tl.float32)
    qo = tl.load(base + hq[:, None] * D + 2 * i[None, :] + 1).to(tl.float32)
    ke = tl.load(base + HQ * D + hk[:, None] * D + 2 * i[None, :]).to(tl.float32)
    ko = tl.load(base + HQ * D + hk[:, None] * D + 2 * i[None, :] + 1).to(tl.float32)
    dd = tl.arange(0, D)
    v = tl.load(base + (HQ + HKV) * D + hk[:, None] * D + dd[None, :]).to(tl.float32)
    loc = tl.load(LOC + tok)
    ks = tl.load(KSC)
    vs = tl.load(VSC)
    if QFP8:
        rq = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + EPS)
    else:
        rq = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / D + EPS) * QK_SCALE
    rk = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / D + EPS)
    qe = qe * rq[:, None]
    qo = qo * rq[:, None]
    ke = ke * rk[:, None]
    ko = ko * rk[:, None]
    if ROPE:
        ang = pos * tl.load(INV_FREQ + i)
        cs = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        qe, qo = qe * cs - qo * sn, qe * sn + qo * cs
        ke, ko = ke * cs - ko * sn, ke * sn + ko * cs
    qbase = QOUT + tok * HQ * D + hq[:, None] * D + 2 * i[None, :]
    tl.store(qbase, qe.to(QOUT.dtype.element_ty))
    tl.store(qbase + 1, qo.to(QOUT.dtype.element_ty))
    kbase = KC + (loc * HKV + hk[:, None]) * D + 2 * i[None, :]
    tl.store(kbase, (ke / ks).to(tl.float8e4nv))
    tl.store(kbase + 1, (ko / ks).to(tl.float8e4nv))
    tl.store(VC + (loc * HKV + hk[:, None]) * D + dd[None, :], (v / vs).to(tl.float8e4nv))


@triton.jit
def _attn_kernel(Q, KC, VC, PT, SEQ, QKV, OUT, WS_O, WS_M, CNT, KSC, VSC,
                 stride_pt, stride_qkv, chunk, nsplit,
                 L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 GROUP: tl.constexpr, KVPC: tl.constexpr, ROWS: tl.constexpr, PAGE: tl.constexpr,
                 WINDOW: tl.constexpr, SM_SCALE: tl.constexpr, BLOCK_N: tl.constexpr, EPS: tl.constexpr,
                 MAXS: tl.constexpr, SB: tl.constexpr, QFP8: tl.constexpr, QK_FACTOR: tl.constexpr,
                 PVFP8: tl.constexpr):
    """One program: request b, KVPC kv heads starting at h0 = program_id(1) * KVPC, one split.
    Rows r = (token j, local kv head hl, head-in-group g); Q is block-diagonal over [ROWS, KVPC*D]
    so one dot against the cache's contiguous KVPC*D-wide rows gives each row its own head."""
    b = tl.program_id(0)
    hgrp = tl.program_id(1)
    s = tl.program_id(2)
    h0 = hgrp * KVPC
    W: tl.constexpr = KVPC * D
    seq = tl.load(SEQ + b)
    r = tl.arange(0, ROWS)
    j = r // (KVPC * GROUP)
    hl = (r // GROUP) % KVPC
    g = r % GROUP
    row_ok = j < L
    head = (h0 + hl) * GROUP + g
    pos = seq + j
    c = tl.arange(0, W)
    cd = c % D
    qmask = row_ok[:, None] & ((c // D)[None, :] == hl[:, None])
    q = tl.load(Q + ((b * L + j[:, None]) * HQ + head[:, None]) * D + cd[None, :], mask=qmask, other=0.0)

    total = seq + L
    if WINDOW > 0:
        base = tl.maximum(seq - WINDOW + 1, 0)
    else:
        base = 0
    lo = base + s * chunk
    hi = tl.minimum(lo + chunk, total)
    ks = tl.load(KSC)
    vs = tl.load(VSC)
    if QFP8:
        qk_scale = SM_SCALE * QK_FACTOR * ks * 1.4426950408889634
    else:
        qk_scale = SM_SCALE * ks * 1.4426950408889634
    # Blocks entirely inside every row's visible range need no mask: t <= seq (all rows see up to
    # seq) and, with a window, t > seq + L - 1 - WINDOW (the last row's lower edge).
    safe_hi = seq + 1
    if WINDOW > 0:
        safe_lo = seq + L - WINDOW
    else:
        safe_lo = 0

    m_i = tl.full((ROWS,), -1.0e30, tl.float32)
    l_i = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, W), tl.float32)
    for t0 in range(lo, hi, BLOCK_N):
        t = t0 + tl.arange(0, BLOCK_N)
        inb = t < hi
        if PAGE == 1:
            krow = tl.load(PT + b * stride_pt + t, mask=inb, other=0)
        else:
            krow = tl.load(PT + b * stride_pt + t // PAGE, mask=inb, other=0) * PAGE + t % PAGE
        koff = (krow.to(tl.int64) * HKV + h0) * D
        if QFP8:
            k = tl.load(KC + koff[:, None] + c[None, :], mask=inb[:, None], other=0.0)
        else:
            k = tl.load(KC + koff[:, None] + c[None, :], mask=inb[:, None], other=0.0).to(tl.bfloat16)
        if PVFP8:
            v = tl.load(VC + koff[:, None] + c[None, :], mask=inb[:, None], other=0.0)
        else:
            v = tl.load(VC + koff[:, None] + c[None, :], mask=inb[:, None], other=0.0).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)) * qk_scale
        if (t0 >= safe_lo) & (t0 + BLOCK_N <= safe_hi) & (t0 + BLOCK_N <= hi):
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            p = tl.exp2(sc - m_new[:, None])
        else:
            vis = inb[None, :] & (t[None, :] <= pos[:, None])
            if WINDOW > 0:
                vis = vis & (t[None, :] > pos[:, None] - WINDOW)
            sc = tl.where(vis, sc, -1.0e30)
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            p = tl.where(vis, tl.exp2(sc - m_new[:, None]), 0.0)
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if PVFP8:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float8e4nv), v)   # p in [0, 1]: E4M3 as FA3
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    # ---- split partials -> workspace; the last split of (b, head group) merges + norm-gate
    pair = b * (HKV // KVPC) + hgrp
    own = (c // D)[None, :] == hl[:, None]   # each row keeps only its own head's D columns
    tl.store(WS_O + ((pair * nsplit + s) * ROWS + r[:, None]) * D + cd[None, :], acc, mask=own,
             cache_modifier=".cg")
    wm = WS_M + (pair * nsplit + s) * ROWS * 2
    tl.store(wm + r * 2, m_i, cache_modifier=".cg")
    tl.store(wm + r * 2 + 1, l_i, cache_modifier=".cg")
    tl.debug_barrier()
    ticket = tl.atomic_add(CNT + pair, 1, sem="acq_rel", scope="gpu")
    if ticket == nsplit - 1:
        d = tl.arange(0, D)
        # vectorized merge: all splits' (m, l) in one tile, partial outputs in pipelined blocks
        sv = tl.arange(0, MAXS)
        sm = sv < nsplit
        mbase = WS_M + (pair * nsplit + sv[:, None]) * ROWS * 2 + r[None, :] * 2
        ms = tl.load(mbase, mask=sm[:, None], other=-1.0e30, cache_modifier=".cg")      # [MAXS, ROWS]
        ls = tl.load(mbase + 1, mask=sm[:, None], other=0.0, cache_modifier=".cg")
        M = tl.max(ms, 0)
        ws = tl.exp2(ms - M[None, :])                                                    # [MAXS, ROWS]
        Lsum = tl.sum(ws * ls, 0)
        O = tl.zeros((ROWS, D), tl.float32)
        sb = tl.arange(0, SB)
        for s0 in tl.range(0, nsplit, SB, num_stages=3):
            si = s0 + sb
            ok = si < nsplit
            wb = tl.exp2(tl.load(WS_M + (pair * nsplit + si[:, None]) * ROWS * 2 + r[None, :] * 2,
                                 mask=ok[:, None], other=-1.0e30, cache_modifier=".cg") - M[None, :])  # [SB, ROWS]
            part = tl.load(WS_O + ((pair * nsplit + si[:, None, None]) * ROWS + r[None, :, None]) * D + d[None, None, :],
                           mask=ok[:, None, None], other=0.0, cache_modifier=".cg")                 # [SB, ROWS, D]
            O += tl.sum(wb[:, :, None] * part, 0)
        a = O / Lsum[:, None] * vs
        rr = tl.rsqrt(tl.sum(a * a, 1) / D + EPS)
        gate = tl.load(QKV + (b * L + j[:, None]) * stride_qkv + (HQ + 2 * HKV) * D + head[:, None] * D + d[None, :],
                       mask=row_ok[:, None], other=0.0).to(tl.float32)
        o = a * rr[:, None] * tl.sigmoid(gate)
        tl.store(OUT + (b * L + j[:, None]) * (HQ * D) + head[:, None] * D + d[None, :], o.to(tl.bfloat16),
                 mask=row_ok[:, None])
        tl.atomic_xchg(CNT + pair, 0, sem="relaxed", scope="gpu")


_BUF = {}
_INV_FREQ = {}

# Tunables per shape class: (kvpc_max, target_ctas, block_n, min_chunk, max_split, warps, stages)
def default_cfg(B, L, hkv, window):
    if L == 1:
        return (2, 528, 64, 64, 256, 4, 3, False)
    return (1, 528, 64, 64, 256, 4, 3, False)


_T = {  # tuned on H100 (tune.py): (B, L, sliding) -> cfg
    (1, 1, False): (1, 264, 128, 32, 512, 4, 4, True),
    (1, 1, True): (1, 1056, 128, 32, 512, 4, 2, True),
    (16, 1, False): (2, 528, 128, 32, 512, 4, 4, True),
    (16, 1, True): (2, 264, 64, 32, 512, 8, 4, True),
    (32, 1, False): (2, 1056, 128, 32, 512, 4, 3, True),
    (32, 1, True): (2, 264, 128, 32, 512, 8, 4, True),
    (1, 8, False): (1, 264, 64, 32, 512, 4, 4, True),
    (1, 8, True): (1, 528, 128, 32, 512, 8, 4, True),
    (16, 8, False): (1, 264, 128, 32, 512, 4, 4, True),
    (16, 8, True): (1, 264, 128, 32, 512, 4, 3, True),
    (32, 8, False): (1, 264, 128, 32, 512, 4, 3, True),
    (32, 8, True): (1, 264, 128, 32, 512, 4, 4, True),
}

CFG_OVERRIDE = {k: tuple(v[:8]) + (True,) for k, v in _T.items()}


def _buffers(device, ws_o, ws_m, pairs):
    key = device
    cur = _BUF.get(key)
    if cur is None or cur[0].numel() < ws_o or cur[1].numel() < ws_m or cur[2].numel() < pairs:
        cur = (torch.empty(max(ws_o, 1 << 20), dtype=torch.float32, device=device),
               torch.empty(max(ws_m, 1 << 16), dtype=torch.float32, device=device),
               torch.zeros(max(pairs, 4096), dtype=torch.int32, device=device))
        _BUF[key] = cur
    return cur


def _inv_freq(theta, d, device):
    key = (theta, d, device)
    if key not in _INV_FREQ:
        _INV_FREQ[key] = (1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))).contiguous()
    return _INV_FREQ[key]


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window, theta = cfg["window"], cfg["rope_theta"]
    group = hq // hkv
    qkv = data["qkv"].view(B * L, -1)
    dev = qkv.device
    tokens = B * L
    rope = bool(theta)
    inv = _inv_freq(theta if rope else 1.0, d, dev)

    ccfg0 = tuple(CFG_OVERRIDE.get((B, L, bool(window)), default_cfg(B, L, hkv, window)))
    qfp8_0 = bool(ccfg0[7]) if len(ccfg0) > 7 else False
    q = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn if qfp8_0 else torch.bfloat16, device=dev)
    _prep_kernel[(tokens,)](qkv, data["positions"], data["cache_loc"], q, data["k_cache"], data["v_cache"], inv,
                            data["k_scale"], data["v_scale"], qkv.stride(0),
                            HQ=hq, HKV=hkv, D=d, ROPE=rope, QK_SCALE=cfg["qk_scale_factor"], EPS=RMS_EPS, QFP8=qfp8_0,
                            num_warps=4)

    # split count from shapes only: span = window (+L) on sliding layers, page-table capacity on full ones
    ccfg = tuple(CFG_OVERRIDE.get((B, L, bool(window)), default_cfg(B, L, hkv, window)))
    kvmax, target, block_n, min_chunk, max_split, warps, stages = ccfg[:7]
    qfp8 = bool(ccfg[7]) if len(ccfg) > 7 else False
    pvfp8 = bool(ccfg[8]) if len(ccfg) > 8 else False
    kvpc = hkv if hkv <= kvmax else 1
    span = (window + L - 1) if window else data["page_table"].shape[1] * page
    pairs = B * (hkv // kvpc)
    want = max(1, min(max_split, target // pairs))
    chunk = max(min_chunk, triton.cdiv(triton.cdiv(span, want), block_n) * block_n)
    nsplit = triton.cdiv(span, chunk)
    rows = max(16, triton.next_power_of_2(L * group * kvpc))
    ws_o, ws_m, cnt = _buffers(dev, pairs * nsplit * rows * d, pairs * nsplit * rows * 2, pairs)
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=dev)
    _attn_kernel[(B, hkv // kvpc, nsplit)](q, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"],
                                          qkv, out, ws_o, ws_m, cnt, data["k_scale"], data["v_scale"],
                                          data["page_table"].stride(0), qkv.stride(0), chunk, nsplit,
                                          L=L, HQ=hq, HKV=hkv, D=d, GROUP=group, KVPC=kvpc, ROWS=rows, PAGE=page,
                                          WINDOW=window, SM_SCALE=cfg["sm_scale"], BLOCK_N=block_n, EPS=RMS_EPS,
                                          MAXS=max(2, triton.next_power_of_2(nsplit)), SB=max(1, min(8, 8192 // (rows * d))),
                                          QFP8=qfp8, QK_FACTOR=cfg["qk_scale_factor"], PVFP8=pvfp8,
                                          num_warps=warps, num_stages=stages)
    return out
