"""Correctness sweep of the CUDA skinny kernel over m, layers and configs."""
import itertools, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import reference as R
import skinny_cuda
from bench import load_module
t0 = time.time(); skinny_cuda.module(); print(f"built in {time.time()-t0:.0f}s", flush=True)
sub = load_module("submission.py", "submission")
weights = R.generate_weights(R.WEIGHTS_SEED)
state = sub.prepare(weights)
bad = 0; total = 0
for m in [1, 3, 8, 9, 16, 17, 32]:
    for layer in R.LAYERS:
        for dist in ["normal", "outliers"]:
            data = R.generate_input(m, layer, 7, dist=dist)
            n, k = data["config"]["n"], data["config"]["k"]
            f32w = layer == "router"
            for r, u, warps, split in itertools.product([1, 2], [1, 2, 4], [2, 4, 8], [1, 4, 16]):
                if not skinny_cuda.valid(m, n, k, r, u, warps, split, f32w):
                    continue
                out = sub.run(data["x"], layer, n, state, ("cuda", r, u, warps, split))
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(data, out, weights)
                total += 1
                if not ok:
                    bad += 1
                    if bad < 15: print("FAIL", m, layer, dist, (r, u, warps, split), msg, flush=True)
            for bk, stages, warps, split in itertools.product([64, 128], [3, 4, 6], [2, 4, 8], [1, 4, 16]):
                if not skinny_cuda.ring_valid(m, n, k, bk, stages, warps, split, f32w):
                    continue
                out = sub.run(data["x"], layer, n, state, ("ring", bk, stages, warps, split))
                torch.cuda.synchronize()
                ok, msg = R.check_implementation(data, out, weights)
                total += 1
                if not ok:
                    bad += 1
                    if bad < 15: print("FAIL ring", m, layer, dist, (bk, stages, warps, split), msg, flush=True)
print(f"{total - bad}/{total} configs pass", flush=True)
