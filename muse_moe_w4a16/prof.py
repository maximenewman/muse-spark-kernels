"""One kernel_fn call under ncu (--profile-from-start off). usage: prof.py <M> [submission.py]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import reference as R
from bench import gen, load_module
M = int(sys.argv[1])
sub = load_module(sys.argv[2] if len(sys.argv) > 2 else "submission.py", "submission")
st = sub.prepare(R.generate_weights(R.WEIGHTS_SEED))
if os.environ.get("NOCOMPUTE"):
    sub.EXTRA_DEFS["NOCOMPUTE"] = 1
if os.environ.get("CFG"):
    sub.CFG_OVERRIDE[M] = tuple(int(v) for v in os.environ["CFG"].split(","))
data = gen(M, seed=100)
for _ in range(3):
    sub.kernel_fn(data, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
sub.kernel_fn(data, st)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
