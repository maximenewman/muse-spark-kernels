import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import reference as R
from bench import Flusher, gen, graph_overhead, load_module
from tune import time_cfg
w = R.generate_weights(R.WEIGHTS_SEED)
sub = load_module("submission.py", "submission")
st = sub.prepare(w)
flush = Flusher(); ov = graph_overhead(flush)
sets = [gen(1, seed=100 + i) for i in range(2)]
for cfg in [None, (8, 6, 16), (8, 3, 2), (8, 6, 16), (16, 4, 8), (8, 6, 16), None]:
    if cfg is None:
        sub.CFG_OVERRIDE.pop(1, None)
    else:
        sub.CFG_OVERRIDE[1] = cfg
    print(cfg, f"{time_cfg(sub, st, sets, flush, ov, 40):.2f}", torch.cuda.memory_allocated() >> 20, "MiB", flush=True)
