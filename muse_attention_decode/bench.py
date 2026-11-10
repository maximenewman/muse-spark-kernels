"""Local stand-in for the okc judge on muse_attention_decode (see ../muse_bf16_proj/bench.py).

  python bench.py                        # 16 tests + 12 scored shapes vs SGLang (baseline.py)
  python bench.py --sub exp/x.py --skip-tests --only decode16
  python bench.py --base-only            # just SGLang's times (sub = baseline)

Each shape: two independent input sets (seeds), a CUDA graph per (side, set); every replay
starts with a cold L2; SGLang and the submission alternate in AB/BA pairs on the same set; a
shape's speedup is the median of paired ratios; a fixed graph-replay cost is subtracted; the
problem speedup is the geomean over loads of SGLang's load time over ours.
"""

import argparse
import importlib.util
import math
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as R  # noqa: E402

CTX = 80000
SCORED = []  # (load, kwargs, weight)
for load, batch, q_len in [("decode 1", 1, 1), ("decode 16", 16, 1), ("decode 32", 32, 1),
                           ("verify 1", 1, 8), ("verify 16", 16, 8), ("verify 32", 32, 8)]:
    spread = "fixed" if batch == 1 else "agentic"
    SCORED.append((load, dict(batch=batch, ctx=CTX, heads="tp8", layer="sliding", page=1, q_len=q_len,
                              spread=spread, dist="soft"), 46))
    SCORED.append((load, dict(batch=batch, ctx=CTX, heads="tp8", layer="full", page=1, q_len=q_len,
                              spread=spread, dist="coherent"), 16))

TESTS = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
    dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=4, ctx=3000, heads="tp8", layer="full"),
    dict(batch=8, ctx=8192, heads="dp", layer="full", spread="mixed"),
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
    dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8),
    dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
    dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="soft"),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
    dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="coherent"),
]


def load_module(path, name):
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


def desc(kw):
    return ", ".join(f"{k} {v}" for k, v in kw.items())


def run_tests(sub, seeds=(1,)):
    ok_all = True
    for kw in TESTS:
        for seed in seeds:
            data = R.generate_input(seed=seed, **kw)
            out = sub.kernel_fn(data)
            raw = R.collect(data, out)
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, raw)
            print(f"  {'PASS' if ok else 'FAIL'} {desc(kw)}: {msg}", flush=True)
            ok_all &= ok
            del data, out, raw
            torch.cuda.empty_cache()
    return ok_all


def check_scored(sub, kw, seed=7):
    data = R.generate_input(seed=seed, **kw)
    out = sub.kernel_fn(data)
    ok, msg = R.check_implementation(data, R.collect(data, out))
    return ok, msg


def time_shape(sub, base, kw, reps, flush, overhead):
    sets = [R.generate_input(seed=100 + i, **kw) for i in range(2)]
    gs = [capture(lambda d=d: sub.kernel_fn(d)) for d in sets]
    gb = [capture(lambda d=d: base.kernel_fn(d)) for d in sets]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--base", default=os.path.join(HERE, "baseline.py"))
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="loads, e.g. decode16 verify1")
    ap.add_argument("--layer", default=None, choices=["sliding", "full"])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name()}  torch {torch.__version__}", flush=True)
    base = load_module(args.base, "baseline")
    sub = base if args.base_only else load_module(args.sub, "submission")
    if not args.skip_tests and not args.base_only:
        print("tests:", flush=True)
        if not run_tests(sub):
            print("FAILED")
            sys.exit(1)

    flush = Flusher()
    overhead = graph_overhead(flush)
    print(f"subtracting {overhead:.2f} µs fixed replay cost\n")
    hw = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
    res = {}
    print(f"{'load':<10} {'layer':<8} {'sglang µs':>10} {'ours µs':>9} {'speedup':>8} {'floor':>7}  check")
    for load, kw, w in SCORED:
        if args.only and load.replace(" ", "") not in args.only:
            continue
        if args.layer and kw["layer"] != args.layer:
            continue
        chk = ""
        if not args.base_only:
            ok, msg = check_scored(sub, kw)
            chk = "ok" if ok else f"FAIL {msg}"
        b, s, sp = time_shape(sub, base, kw, args.reps, flush, overhead)
        res[(load, kw["layer"])] = (b, s, w)
        floor = R.speed_of_light_ns(hw=hw, **kw) / 1e3
        print(f"{load:<10} {kw['layer']:<8} {b:>10.2f} {s:>9.2f} {sp:>7.3f}x {floor:>6.1f}  {chk}", flush=True)
    loads = sorted({l for l, _ in res}, key=lambda l: [x[0] for x in SCORED].index(l))
    logs = []
    print()
    for load in loads:
        parts = [v for (l, _), v in res.items() if l == load]
        if len(parts) < 2:
            continue
        bt = sum(w * b for b, s, w in parts)
        st = sum(w * s for b, s, w in parts)
        logs.append(math.log(bt / st))
        print(f"  {load:<10} sglang {bt:9.1f} µs/step  ours {st:9.1f}  speedup {bt / st:.3f}x")
    if len(logs) == 6:
        print(f"\nproblem speedup (geomean over loads): {math.exp(sum(logs) / len(logs)):.4f}x")


if __name__ == "__main__":
    main()
