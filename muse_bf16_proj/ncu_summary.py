"""Summarize ncu reports: time, DRAM TB/s, warps, top stall reasons.  usage: ncu_summary.py a.ncu-rep ..."""
import csv, io, subprocess, sys

KEYS = {"gpu__time_duration.sum": "us", "dram__bytes_read.sum": "MB",
        "smsp__warps_active.avg.per_cycle_active": "warps/sched",
        "smsp__warps_eligible.avg.per_cycle_active": "eligible/sched",
        "launch__registers_per_thread": "regs", "launch__grid_size": "grid", "launch__block_size": "block"}


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


for rep in sys.argv[1:]:
    out = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    h, u, v = rows[0], rows[1], rows[2]
    d = {n: (num(v[i]), u[i]) for i, n in enumerate(h)}
    t = d["gpu__time_duration.sum"][0] * (1e-3 if d["gpu__time_duration.sum"][1] == "ns" else 1)
    mb = d["dram__bytes_read.sum"][0] * {"Mbyte": 1, "Gbyte": 1e3, "Kbyte": 1e-3}.get(d["dram__bytes_read.sum"][1], 1)
    stalls = sorted(((val, n[len("smsp__average_warps_issue_stalled_"):-len("_per_issue_active.ratio")])
                     for n, (val, _) in d.items()
                     if n.startswith("smsp__average_warps_issue_stalled_") and val is not None), reverse=True)
    name = rep.split("/")[-1].replace(".ncu-rep", "")
    extra = " ".join(f"{lab}={d[k][0]:g}" for k, lab in KEYS.items() if k in d and d[k][0] is not None and lab not in ("us", "MB"))
    print(f"{name:16s} {t:6.2f} us  {mb / t / 1e6:5.2f} TB/s  {extra}")
    print(f"{'':16s} stalls/issue: " + "  ".join(f"{n}={x:.1f}" for x, n in stalls[:5]))
