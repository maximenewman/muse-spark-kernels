"""Local stand-in for the okc judge on muse_moe_w4a16 (see ../muse_attention_decode/bench.py).

  python bench.py                       # 5 tests + 6 scored shapes vs SGLang's Marlin path
  python bench.py --sub exp/x.py --skip-tests --only decode16
  python bench.py --base-only

Each shape: two input sets (seeds), a CUDA graph per (side, set); every replay starts with a
cold L2; SGLang and the submission alternate in AB/BA pairs; a shape's speedup is the median
of paired ratios; a fixed graph-replay cost is subtracted; each load has one shape, so the
problem speedup is the geomean of the six shape speedups.
"""

import argparse
import importlib.util
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as R  # noqa: E402

SCORED = [("decode 1", 1), ("decode 16", 16), ("decode 32", 32),
          ("verify 1", 8), ("verify 16", 128), ("verify 32", 256)]
TESTS = [dict(bs=4), dict(bs=33), dict(bs=257), dict(bs=128, dist="skewed"), dict(bs=1)]


def load_module(path, name):
    if name == "baseline":
        import sgl_stubs  # serving-only sglang deps the MoE baseline never calls
        sgl_stubs.install()
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Flusher:
    def __init__(self, nbytes=512 << 20):
        self.buf = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda").fill_(1.0)
        self.out = torch.empty((), dtype=torch.float32, device="cuda")

    def __call__(self):
        torch.sum(self.buf, dim=0, out=self.out)


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def graph_overhead(flush, reps=200):
    t = torch.zeros(1, device="cuda")
    g = capture(lambda: t.add_(1))
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        flush()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)


def gen(bs, seed, dist="normal"):
    return R.generate_input(bs=bs, sharding="ep8", seed=seed, dist=dist)


def run_tests(sub, state, weights):
    ok_all = True
    for kw in TESTS:
        data = gen(seed=1, **kw)
        out = sub.kernel_fn(data, state)
        torch.cuda.synchronize()
        ok, msg = R.check_implementation(data, out, weights)
        print(f"  {'PASS' if ok else 'FAIL'} {kw}: {msg}", flush=True)
        ok_all &= ok
    return ok_all


def time_shape(sub, ss, base, bs_state, bs, reps, flush, overhead):
    sets = [gen(bs, seed=100 + i) for i in range(2)]
    gs = [capture(lambda d=d: sub.kernel_fn(d, ss)) for d in sets]
    gb = [capture(lambda d=d: base.kernel_fn(d, bs_state)) for d in sets]
    ev = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(2)]
          for _ in range(2 * reps)]

    def one(g, a, b):
        flush()
        a.record()
        g.replay()
        b.record()

    for i in range(6):
        one(gs[i % 2], *ev[0][0])
        one(gb[i % 2], *ev[0][1])
    torch.cuda.synchronize()
    for r in range(2 * reps):
        k = (r // 2) % 2
        if r % 2 == 0:
            one(gb[k], *ev[r][1]); one(gs[k], *ev[r][0])
        else:
            one(gs[k], *ev[r][0]); one(gb[k], *ev[r][1])
    torch.cuda.synchronize()
    ts = [max(ev[r][0][0].elapsed_time(ev[r][0][1]) * 1e3 - overhead, 0.05) for r in range(2 * reps)]
    tb = [max(ev[r][1][0].elapsed_time(ev[r][1][1]) * 1e3 - overhead, 0.05) for r in range(2 * reps)]
    ratios = [(tb[2 * i] + tb[2 * i + 1]) / (ts[2 * i] + ts[2 * i + 1]) for i in range(reps)]
    sp = statistics.median(ratios)
    bt = statistics.median(tb)
    del sets, gs, gb
    torch.cuda.empty_cache()
    return bt, bt / sp, sp


def routing_stats(bs, seeds=range(100, 104)):
    loads, distinct, mx = [], [], []
    for s in seeds:
        ids = gen(bs, s)["topk_ids"]
        loc = ids[ids < 32]
        c = torch.bincount(loc.long(), minlength=32)
        loads.append(int(c.sum())); distinct.append(int((c > 0).sum())); mx.append(int(c.max()))
    return statistics.mean(loads), statistics.mean(distinct), max(mx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--base", default=os.path.join(HERE, "baseline.py"))
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="loads, e.g. decode16 verify1")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name()}  torch {torch.__version__}", flush=True)
    t0 = time.time()
    weights = R.generate_weights(R.WEIGHTS_SEED)
    torch.cuda.synchronize()
    print(f"weights in {time.time() - t0:.0f}s", flush=True)
    if args.stats:
        for load, bs in SCORED:
            print(f"  {load:<10} bs {bs:4d}: local routes, distinct experts, max per expert = {routing_stats(bs)}")
    base = load_module(args.base, "baseline")
    bstate = base.prepare(weights)
    sub = base if args.base_only else load_module(args.sub, "submission")
    t0 = time.time()
    sstate = bstate if args.base_only else sub.prepare(weights)
    torch.cuda.synchronize()
    print(f"prepare in {time.time() - t0:.1f}s", flush=True)
    if not args.skip_tests and not args.base_only:
        print("tests:", flush=True)
        if not run_tests(sub, sstate, weights):
            print("FAILED")
            sys.exit(1)

    flush = Flusher()
    overhead = graph_overhead(flush)
    print(f"subtracting {overhead:.2f} µs fixed replay cost\n")
    hw = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
    logs = []
    print(f"{'load':<10} {'bs':>4} {'sglang µs':>10} {'ours µs':>9} {'speedup':>8} {'floor':>7}  check")
    for load, bs in SCORED:
        if args.only and load.replace(" ", "") not in args.only:
            continue
        chk = ""
        if not args.base_only:
            d = gen(bs, seed=7)
            ok, msg = R.check_implementation(d, sub.kernel_fn(d, sstate), weights)
            chk = "ok" if ok else f"FAIL {msg}"
        b, s, sp = time_shape(sub, sstate, base, bstate, bs, args.reps, flush, overhead)
        logs.append(math.log(sp))
        load_, distinct, _ = routing_stats(bs, seeds=[100, 101])
        floor = R.speed_of_light_ns(bs, "ep8", hw, router_load=load_, router_distinct=distinct) / 1e3
        print(f"{load:<10} {bs:>4} {b:>10.2f} {s:>9.2f} {sp:>7.3f}x {floor:>6.1f}  {chk}", flush=True)
    if len(logs) == 6:
        print(f"\nproblem speedup (geomean over loads): {math.exp(sum(logs) / len(logs)):.4f}x")


if __name__ == "__main__":
    main()
