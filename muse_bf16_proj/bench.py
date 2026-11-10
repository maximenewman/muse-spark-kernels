"""Local stand-in for the okc judge on muse_bf16_proj.

  python bench.py                       # tests + every scored shape, vs the SGLang baseline
  python bench.py --sub exp/foo.py      # another submission file
  python bench.py --only router --m 1 16
  python bench.py --skip-tests --reps 400

Mirrors TASK.md: each shape is captured as a CUDA graph; every replay starts with a cold
L2 (a buffer bigger than L2 is read first); baseline and submission take turns on the same
fresh inputs, in pairs with the order swapped; a shape's speedup is the median of paired
ratios; a fixed graph-replay cost is subtracted from both sides; the problem speedup is the
geometric mean over loads of SGLang's load time over ours.

baseline.py imports sglang only for _bf16_gemm_dispatch_impl, which on H200 (default
backend) is F.linear; without sglang installed we use F.linear directly.
"""

import argparse
import importlib.util
import math
import os
import statistics
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference as R  # noqa: E402

LOADS = {"decode 1": 1, "decode 16": 16, "decode 32": 32, "verify 1": 8, "verify 16": 128, "verify 32": 256}
LAYER_ORDER = ["qkv_gate", "o", "pooler", "unpooler", "router"]
WEIGHT = 62
TESTS = [
    (1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"),
    (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"),
    (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers"),
]


def load_module(path, name):
    try:
        import sglang.srt.layers.quantization.unquant  # noqa: F401
    except Exception:
        if name == "baseline":  # stub the one sglang import baseline.py needs
            for mod in ["sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.quantization"]:
                sys.modules.setdefault(mod, types.ModuleType(mod))
            stub = types.ModuleType("sglang.srt.layers.quantization.unquant")
            stub._bf16_gemm_dispatch_impl = lambda x, w, b: F.linear(x, w, b)
            sys.modules["sglang.srt.layers.quantization.unquant"] = stub
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Flusher:
    """Reads a buffer larger than L2 so the next replay starts cold."""

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
        out = fn()
    torch.cuda.synchronize()
    return g, out


def time_replay(g, flush, x_static, x_new, start, end):
    x_static.copy_(x_new)
    flush()
    start.record()
    g.replay()
    end.record()


def graph_overhead(flush, reps=200):
    """Replay time of a graph holding one trivial kernel: the fixed cost to subtract."""
    t = torch.zeros(1, device="cuda")
    g, _ = capture(lambda: t.add_(1))
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        flush()
        a.record()
        g.replay()
        b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)


def make_inputs(m, layer, count, seed0=1000, dist="normal"):
    return [R.generate_input(m, layer, seed0 + i, dist=dist) for i in range(count)]


def run_tests(sub, s_state, weights, extra_shapes):
    ok_all = True
    cases = [(m, l, d) for m, l, d in TESTS] + [(m, l, "normal") for m, l in extra_shapes]
    for m, layer, dist in cases:
        for seed in (1, 2):
            data = R.generate_input(m, layer, seed, dist=dist)
            out = sub.kernel_fn(data, s_state)
            torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, out, weights)
            if not ok or seed == 1:
                print(f"  {'PASS' if ok else 'FAIL'} m={m:<5} {layer:<9} {dist:<8} {msg}")
            ok_all &= ok
    return ok_all


def time_shape(sub, s_state, base, b_state, m, layer, reps, flush, overhead):
    inputs = make_inputs(m, layer, 16)
    data_s = {"x": inputs[0]["x"].clone(), "config": inputs[0]["config"]}
    data_b = {"x": inputs[0]["x"].clone(), "config": inputs[0]["config"]}
    gs, _ = capture(lambda: sub.kernel_fn(data_s, s_state))
    gb, _ = capture(lambda: base.kernel_fn(data_b, b_state))
    evs = [[(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(2)]
           for _ in range(2 * reps)]
    # warm both graphs
    for i in range(10):
        time_replay(gs, flush, data_s["x"], inputs[i % 16]["x"], *evs[0][0])
        time_replay(gb, flush, data_b["x"], inputs[i % 16]["x"], *evs[0][1])
    torch.cuda.synchronize()
    for r in range(2 * reps):
        x_new = inputs[r % 16]["x"]
        order = [(gb, data_b, 1), (gs, data_s, 0)] if r % 2 == 0 else [(gs, data_s, 0), (gb, data_b, 1)]
        for g, d, slot in order:
            time_replay(g, flush, d["x"], x_new, *evs[r][slot])
    torch.cuda.synchronize()
    ts = [max(evs[r][0][0].elapsed_time(evs[r][0][1]) * 1e3 - overhead, 0.05) for r in range(2 * reps)]
    tb = [max(evs[r][1][0].elapsed_time(evs[r][1][1]) * 1e3 - overhead, 0.05) for r in range(2 * reps)]
    ratios = [(tb[2 * i] + tb[2 * i + 1]) / (ts[2 * i] + ts[2 * i + 1]) for i in range(reps)]
    speedup = statistics.median(ratios)
    base_t = statistics.median(tb)
    return base_t, base_t / speedup, speedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(HERE, "submission.py"))
    ap.add_argument("--base", default=os.path.join(HERE, "baseline.py"))
    ap.add_argument("--only", nargs="*", default=None, help="layers to time")
    ap.add_argument("--m", nargs="*", type=int, default=None, help="m values to time")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--overhead", type=float, default=None, help="µs to subtract (default: measured)")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"GPU {torch.cuda.get_device_name()}  torch {torch.__version__}")
    weights = R.generate_weights(R.WEIGHTS_SEED)
    sub = load_module(args.sub, "submission")
    base = load_module(args.base, "baseline")
    s_state = sub.prepare({k: v.clone() for k, v in weights.items()})
    b_state = base.prepare({k: v.clone() for k, v in weights.items()})

    layers = args.only or LAYER_ORDER
    ms = args.m or sorted(set(LOADS.values()))
    shapes = [(m, l) for m in ms for l in layers]

    if not args.skip_tests:
        print("tests (+ the timed shapes):")
        if not run_tests(sub, s_state, weights, shapes):
            print("FAILED")
            sys.exit(1)

    flush = Flusher()
    overhead = args.overhead if args.overhead is not None else graph_overhead(flush)
    print(f"subtracting {overhead:.2f} µs fixed replay cost\n")
    print(f"{'m':>4} {'layer':<9} {'sglang µs':>10} {'ours µs':>9} {'speedup':>8} {'floor':>7}")
    res = {}
    hw = {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
    for m, layer in shapes:
        b, s, sp = time_shape(sub, s_state, base, b_state, m, layer, args.reps, flush, overhead)
        res[(m, layer)] = (b, s)
        floor = R.speed_of_light_ns(m, layer, hw) / 1e3
        print(f"{m:>4} {layer:<9} {b:>10.2f} {s:>9.2f} {sp:>7.3f}x {floor:>6.1f}", flush=True)

    full = [name for name, m in LOADS.items() if all((m, l) in res for l in LAYER_ORDER)]
    if full:
        print()
        logs = []
        for name in full:
            m = LOADS[name]
            bt = sum(WEIGHT * res[(m, l)][0] for l in LAYER_ORDER)
            st = sum(WEIGHT * res[(m, l)][1] for l in LAYER_ORDER)
            logs.append(math.log(bt / st))
            print(f"  {name:<10} sglang {bt / 1e3:8.1f} ms/step  ours {st / 1e3:8.1f}  speedup {bt / st:.3f}x")
        if len(full) == len(LOADS):
            print(f"\nproblem speedup (geomean over loads): {math.exp(sum(logs) / len(logs)):.4f}x")


if __name__ == "__main__":
    main()
