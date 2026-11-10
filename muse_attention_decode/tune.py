"""Sweep submission.CFG_OVERRIDE per scored shape (graph replays, cold L2).

  python tune.py [--only decode1 verify32] [--layer sliding|full] [--out tuned.json]
cfg = (kvpc_max, target_ctas, block_n, min_chunk, max_split, warps, stages)
"""
import argparse
import itertools
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as R  # noqa: E402
from bench import SCORED, Flusher, capture, graph_overhead, load_module  # noqa: E402


def candidates(kw):
    out = []
    for kv, target, bn, warps, stages in itertools.product([1, 2], [132, 264, 528, 1056], [32, 64, 128], [4, 8], [2, 3, 4]):
        if kw["q_len"] == 8 and kv == 2 and warps == 4:
            continue  # 128 rows x 128 cols: needs 8 warps
        for qf in (False, True):
            out.append((kv, target, bn, 32, 512, warps, stages, qf))
    return out


def key(kw):
    return (kw["batch"], kw["q_len"], kw["layer"] == "sliding")


def _compile_worker(job):
    sub_path, kw, cfgs = job
    sub = load_module(sub_path, "submission")
    data = R.generate_input(seed=3, **{**kw, "ctx": min(kw["ctx"], 4096)})  # small ctx: same kernels
    bad = 0
    for cfg in cfgs:
        sub.CFG_OVERRIDE[key(kw)] = tuple(cfg)
        try:
            sub.kernel_fn(data)
        except Exception:
            bad += 1
    torch.cuda.synchronize()
    return bad


def time_cfg(sub, kw, data, cfg, flush, overhead, reps):
    sub.CFG_OVERRIDE[key(kw)] = tuple(cfg)
    g = capture(lambda: sub.kernel_fn(data))
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        flush()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    del g
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev) - overhead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--layer", default=None)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "tuned.json"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    shapes = [(l, kw) for l, kw, _ in SCORED if (not args.only or l.replace(" ", "") in args.only)
              and (not args.layer or kw["layer"] == args.layer)]

    import multiprocessing as mp
    t0 = time.time()
    jobs = [(args.sub, kw, candidates(kw)[i::args.workers]) for _, kw in shapes for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        bad = sum(pool.map(_compile_worker, jobs))
    print(f"precompiled in {time.time() - t0:.0f}s ({bad} failed)", flush=True)

    sub = load_module(args.sub, "submission")
    flush = Flusher()
    overhead = graph_overhead(flush)
    tuned = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for load, kw in shapes:
        data = R.generate_input(seed=11, **kw)
        res = []
        for cfg in candidates(kw):
            try:
                res.append((time_cfg(sub, kw, data, cfg, flush, overhead, args.reps), cfg))
            except Exception as e:
                print(f"  skip {cfg}: {type(e).__name__} {str(e)[:60]}", flush=True)
        res.sort()
        final = sorted((time_cfg(sub, kw, data, c, flush, overhead, args.reps * 4), c) for _, c in res[:4])
        ok, msg = True, ""
        sub.CFG_OVERRIDE[key(kw)] = tuple(final[0][1])
        out = sub.kernel_fn(data)
        ok, msg = R.check_implementation(data, R.collect(data, out))
        print(f"{load:<10} {kw['layer']:<8} best {final[0][0]:8.2f} µs {final[0][1]}  check {'ok' if ok else msg}", flush=True)
        for t, c in final[1:]:
            print(f"{'':30s} {t:8.2f} µs {c}", flush=True)
        tuned[",".join(map(str, key(kw)))] = list(final[0][1])
        json.dump(tuned, open(args.out, "w"), indent=1)
        del data
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
