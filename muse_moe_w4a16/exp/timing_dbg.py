import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import reference as R
from bench import Flusher, capture, gen, graph_overhead, load_module
from tune import time_cfg
w = R.generate_weights(R.WEIGHTS_SEED)
sub = load_module("submission.py", "submission")
st = sub.prepare(w)
flush = Flusher(); ov = graph_overhead(flush)
print("overhead", ov)
for M in (1, 32):
    sets = [gen(M, seed=100 + i) for i in range(2)]
    print(M, "tune-style", time_cfg(sub, st, sets, flush, ov, 40))
    sub.CFG_OVERRIDE[M] = sub._cfg(M)
    print(M, "tune-style w/ override", time_cfg(sub, st, sets, flush, ov, 40))
    sub.CFG_OVERRIDE.pop(M)
    g = capture(lambda: sub.kernel_fn(sets[0], st))
    ts = []
    for _ in range(40):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        flush(); a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 - ov)
    print(M, "single graph", statistics.median(ts))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        flush(); g.replay(); torch.cuda.synchronize()
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
