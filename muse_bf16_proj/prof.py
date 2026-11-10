"""Run one kernel config under ncu: warm up untimed, then one call inside cudaProfilerStart/Stop.

  ncu --profile-from-start off ... python prof.py <layer> <m> <kind> <cfg...> [--layout v2|v3]
kind: triton (cfg = submission CONFIGS tuple as JSON) | cuda (r u warps split)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import reference as R
import skinny_cuda
from bench import load_module

layer, m, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
layout = "v3"
if "--layout" in sys.argv:
    layout = sys.argv[sys.argv.index("--layout") + 1]
sub = load_module("submission.py", "submission")
weights = R.generate_weights(R.WEIGHTS_SEED)
state = sub.prepare(weights)
data = R.generate_input(m, layer, 3)
x, n = data["x"], data["config"]["n"]
if kind == "cuda":
    r, u, warps, split = map(int, sys.argv[4:8])
    w = state["router"] if layer == "router" else state[layer]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device="cuda")
    fn = lambda: skinny_cuda.run(x, w, out, state["ws"], state["cnt"], r, u, warps, split, layout)
elif kind == "ring":
    bk, stages, warps, split = map(int, sys.argv[4:8])
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    fn = lambda: skinny_cuda.run_ring(x, state[layer], out, state["ws"], state["cnt"], bk, stages, warps, split)
else:
    cfg = tuple(json.loads(sys.argv[4]))
    fn = lambda: sub.run(x, layer, n, state, cfg)
for _ in range(5):
    o = fn()
torch.cuda.synchronize()
ok, msg = R.check_implementation(data, o, weights)
print("check:", ok, msg, flush=True)
torch.cuda.cudart().cudaProfilerStart()
fn()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
