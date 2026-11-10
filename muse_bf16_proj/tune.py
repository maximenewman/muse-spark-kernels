"""Sweep submission.py's kernel configs per scored shape (graph replays, cold L2).

  python tune.py                         # all scored shapes
  python tune.py --only router --m 1 8   # a subset
Writes tuned.json ({"layer,m": [cfg...]}) and prints a CONFIGS block to paste into submission.py.
"""

import argparse
import itertools
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as R  # noqa: E402
import skinny_cuda  # noqa: E402

args_cuda_only = False
from bench import LAYER_ORDER, LOADS, Flusher, capture, graph_overhead, load_module  # noqa: E402


def candidates(m, n, k, sms):
    """A focused space: CTA counts between ~1 and ~6 waves of SMs, few stage/warp options."""
    out = []
    if m <= 32:  # hand-written CUDA kernel (skinny_cuda.py)
        f32w = n == 256 and k == 8192  # the router
        for r, u, warps in itertools.product([1, 2], [1, 2, 4], [2, 4, 8]):
            for split in [1, 2, 4, 8, 16, 32]:
                ctas = (n // (16 * r)) * split
                if skinny_cuda.valid(m, n, k, r, u, warps, split, f32w) and sms <= ctas <= 16 * sms:
                    out.append(("cuda", r, u, warps, split))
    if m <= 32 and not (n == 256 and k == 8192):  # v4: cp.async smem ring (bf16 only)
        for bk, stages, warps in itertools.product([64, 128], [3, 4, 6], [2, 4, 8]):
            for split in [1, 2, 4, 8, 16]:
                ctas = (n // (16 * warps)) * split
                if skinny_cuda.ring_valid(m, n, k, bk, stages, warps, split, False) and sms // 2 <= ctas <= 8 * sms:
                    out.append(("ring", bk, stages, warps, split))
    if args_cuda_only:
        return out
    if m == 1:
        for bn, bk, warps, stages in itertools.product([4, 8, 16], [256, 512, 1024], [4, 8], [1, 3]):
            if n % bn or not 2048 <= bn * bk <= 8192:
                continue
            for split in [1, 2, 4, 8, 16, 32]:
                ctas = (n // bn) * split
                if k % (split * bk) or not sms <= ctas <= 6 * sms:
                    continue
                out.append(("gemv", bn, 1, bk, split, warps, stages))
        return out
    if m > 32:  # standard-orientation wgmma kernel
        for bm, bn, bk, warps, stages in itertools.product([64, 128] if m <= 128 else [128, 256], [64, 128, 256],
                                                           [64, 128], [4, 8], [3, 4]):
            if n % bn or bm * bn > 128 * 256 or stages * (bm * bk + (2 if n == 256 else 1) * bn * bk) * 2 > 220 * 1024:
                continue
            tiles = (n // bn) * -(-m // bm)
            for split in [1, 2, 4, 8, 16]:
                if k % (split * bk) or not 32 <= tiles * split <= 4 * sms:
                    continue
                if split > 1 and tiles * split * bm * bn * 4 > (64 << 20):
                    continue
                out.append(("mm", bn, bm, bk, split, warps, stages))
    if m <= 32:
        bms, bns, bks, warpss, stagess = [16 if m <= 16 else 32], [16, 32, 64], [128, 256], [4], [3, 5]
    else:
        bms = [64, 128] if m <= 128 else [128, 256]
        bns, bks, warpss, stagess = [64, 128], [64, 128], [4, 8], [3, 4]
    for bm, bn, bk, warps, stages in itertools.product(bms, bns, bks, warpss, stagess):
        if n % bn or bn * bm > 128 * 256:
            continue
        hilo = 2 if n == 256 else 1  # router: hi and lo tiles
        if stages * (hilo * bn * bk + bk * bm) * 2 > 220 * 1024:
            continue
        tiles = (n // bn) * -(-m // bm)
        for split in [1, 2, 4, 8, 16, 32]:
            if k % (split * bk) or not sms // 2 <= tiles * split <= 6 * sms:
                continue
            out.append(("gemm", bn, bm, bk, split, warps, stages))
    return out


def time_cfg(sub, state, inputs, layer, n, cfg, flush, overhead, reps):
    data = {"x": inputs[0]["x"].clone()}
    g, _ = capture(lambda: sub.run(data["x"], layer, n, state, cfg))
    ts = []
    for r in range(reps):
        data["x"].copy_(inputs[r % len(inputs)]["x"])
        flush()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        ts.append((a, b))
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ts) - overhead


def _compile_worker(job):
    """Run each (layer, m, cfg) once so Triton's on-disk cache holds its binary."""
    sub_path, items = job
    sub = load_module(sub_path, "submission")
    weights = R.generate_weights(R.WEIGHTS_SEED)
    state = sub.prepare(weights)
    bad = 0
    for layer, m, cfg in items:
        try:
            x = torch.randn(m, R.LAYERS[layer][1], device="cuda", dtype=torch.bfloat16)
            sub.run(x, layer, R.out_features(layer, m), state, tuple(cfg))
        except Exception:
            bad += 1
    torch.cuda.synchronize()
    return len(items), bad


def precompile(sub_path, items, workers):
    import multiprocessing as mp
    import time
    t0 = time.time()
    chunks = [(sub_path, items[i::workers]) for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        done = pool.map(_compile_worker, chunks)
    print(f"precompiled {sum(d for d, _ in done)} configs ({sum(b for _, b in done)} failed) "
          f"in {time.time() - t0:.0f} s with {workers} workers", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--m", nargs="*", type=int, default=None)
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "tuned.json"))
    ap.add_argument("--cuda-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    global args_cuda_only
    args_cuda_only = args.cuda_only
    ms = args.m or sorted(set(LOADS.values()))
    layers = args.only or LAYER_ORDER
    sms0 = 132
    items = [(l, m, c) for m in ms for l in layers
             for c in candidates(m, R.out_features(l, m), R.LAYERS[l][1], sms0)]
    skinny_cuda.module()  # build the CUDA extension once, before workers load it
    if args.workers > 1:
        precompile(args.sub, items, args.workers)

    weights = R.generate_weights(R.WEIGHTS_SEED)
    sub = load_module(args.sub, "submission")
    state = sub.prepare(weights)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    flush = Flusher()
    overhead = graph_overhead(flush)
    print(f"GPU {torch.cuda.get_device_name()}  overhead {overhead:.2f} µs", flush=True)

    tuned = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for m in ms:
        for layer in layers:
            n, k = R.out_features(layer, m), R.LAYERS[layer][1]
            inputs = [R.generate_input(m, layer, 100 + i) for i in range(8)]
            results = []
            default = sub.pick_config(layer, m, n, k)
            for cfg in [default] + [c for c in candidates(m, n, k, sms) if c != default]:
                try:
                    out = sub.run(inputs[0]["x"], layer, n, state, cfg)
                    torch.cuda.synchronize()
                    ok, _ = R.check_implementation(inputs[0], out, weights)
                    if not ok:
                        print(f"  wrong: {cfg}", flush=True)
                        continue
                    t = time_cfg(sub, state, inputs, layer, n, cfg, flush, overhead, args.reps)
                    results.append((t, cfg))
                except Exception as e:  # out of resources etc.
                    print(f"  skip {cfg}: {type(e).__name__} {str(e)[:80]}", flush=True)
            default_t = results[0][0] if results and results[0][1] == default else float("nan")
            results.sort()
            # re-time the top few more carefully
            final = sorted((time_cfg(sub, state, inputs, layer, n, c, flush, overhead, args.reps * 5), c)
                           for _, c in results[: args.top])
            floor = R.speed_of_light_ns(m, layer, {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}) / 1e3
            print(f"m={m:<4} {layer:<9} floor {floor:5.1f}  default {default_t:6.2f} µs", flush=True)
            for t, c in final:
                print(f"    {t:7.2f} µs  {c}", flush=True)
            tuned[f"{layer},{m}"] = list(final[0][1])
            json.dump(tuned, open(args.out, "w"), indent=1)

    print("\nCONFIGS = {")
    for key, cfg in tuned.items():
        layer, m = key.split(",")
        print(f"    ({layer!r}, {m}): {tuple(cfg)!r},")
    print("}")


if __name__ == "__main__":
    main()
