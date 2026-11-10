import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import reference as R
from bench import gen, load_module
w = R.generate_weights(R.WEIGHTS_SEED)
sub = load_module("submission.py", "submission")
st = sub.prepare(w)
for M, dist in [(1, "normal"), (33, "normal"), (128, "skewed"), (257, "normal")]:
    for cfg in [(8, 3, 8, 1, 1, 2), (8, 3, 8, 2, 2, 2), (16, 2, 16, 1, 1, 8), (32, 2, 16, 1, 1, 8), (8, 3, 8, 1, 1, 4)]:
        sub.CFG_OVERRIDE[M] = cfg
        d = gen(M, seed=3, dist=dist)
        ok, msg = R.check_implementation(d, sub.kernel_fn(d, st), w)
        print(M, dist, cfg, "ok" if ok else "FAIL", msg, flush=True)
