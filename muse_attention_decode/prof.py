"""One kernel_fn call under ncu (--profile-from-start off). usage: prof.py <load> <layer>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import reference as R
from bench import SCORED, load_module
load, layer = sys.argv[1].replace("_", " "), sys.argv[2]
kw = next(k for l, k, _ in SCORED if l == load and k["layer"] == layer)
sub = load_module(sys.argv[3] if len(sys.argv) > 3 else "submission.py", "submission")
data = R.generate_input(seed=5, **kw)
for _ in range(3):
    out = sub.kernel_fn(data)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
sub.kernel_fn(data)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
