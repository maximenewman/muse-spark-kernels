"""Sweep submission.CFG_OVERRIDE[M] = (tokens per item, stages, grid slots) per scored M.

  python tune.py [--m 1 16] [--out tuned.json]
Times our graph replays only (cold L2, median of 2 input sets), checks the winner.
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
from bench import SCORED, Flusher, capture, gen, graph_overhead, load_module  # noqa: E402


def time_cfg(sub, st, sets, flush, overhead, reps):
    gs = [capture(lambda d=d: sub.kernel_fn(d, st)) for d in sets]
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for i in range(4):  # warm
        flush(); gs[i % len(gs)].replay()
    for r, (a, b) in enumerate(ev):  # queue in chunks of 40 (longer queues inflate event times)
        flush()
        a.record()
        gs[r % len(gs)].replay()
        b.record()
        if r % 40 == 39:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    del gs
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev) - overhead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--m", type=int, nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "tuned.json"))
    args = ap.parse_args()
    weights = R.generate_weights(R.WEIGHTS_SEED)
    sub = load_module(args.sub, "submission")
    st = sub.prepare(weights)
    flush = Flusher()
    overhead = graph_overhead(flush)
    tuned = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for load, M in SCORED:
        if args.m and M not in args.m:
            continue
        sets = [gen(M, seed=100 + i) for i in range(2)]
        res = []
        for tb, nst, slots in itertools.product([8, 16, 32, 64], [3, 4, 5, 6, 8], [2, 4, 8, 16, 32]):
            if sub._smem(tb, nst) > 220 * 1024:
                continue
            sub.CFG_OVERRIDE[M] = (tb, nst, slots)
            try:
                res.append((time_cfg(sub, st, sets, flush, overhead, args.reps), (tb, nst, slots)))
                if args.verbose:
                    print(f"    {res[-1][1]} {res[-1][0]:.2f}", flush=True)
            except Exception as e:
                print(f"  skip {(tb, nst, slots)}: {type(e).__name__} {str(e)[:80]}", flush=True)
        res.sort()
        final = []
        for _, c in res[:5]:
            sub.CFG_OVERRIDE[M] = c
            final.append((time_cfg(sub, st, sets, flush, overhead, args.reps * 4), c))
        final.sort()
        sub.CFG_OVERRIDE[M] = final[0][1]
        d = gen(M, seed=7)
        ok, msg = R.check_implementation(d, sub.kernel_fn(d, st), weights)
        print(f"{load:<10} M {M:4d} best {final[0][0]:8.2f} µs {final[0][1]}  {'ok' if ok else msg}", flush=True)
        for t, c in final[1:]:
            print(f"{'':22s} {t:8.2f} µs {c}", flush=True)
        tuned[str(M)] = list(final[0][1])
        json.dump(tuned, open(args.out, "w"), indent=1)
        del sets
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
