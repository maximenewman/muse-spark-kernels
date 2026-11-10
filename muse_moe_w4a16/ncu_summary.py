"""Summarize ncu reports, one line pair per kernel: time, DRAM GB/s, occupancy, top stalls.

  python ncu_summary.py ncu/*.ncu-rep
"""
import csv
import io
import subprocess
import sys


def num(x):
    try:
        return float(x.replace(",", ""))
    except ValueError:
        return None


def to_us(val, unit):
    return val * {"ns": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}.get(unit, 1.0)


def to_mb(val, unit):
    return val * {"byte": 1e-6, "Kbyte": 1e-3, "Mbyte": 1.0, "Gbyte": 1e3}.get(unit, 1.0)


def report(rep, h, u, v):
    d = {n: (num(v[i]), u[i]) for i, n in enumerate(h)}
    t = to_us(*d["gpu__time_duration.sum"])
    mb = to_mb(*d["dram__bytes_read.sum"]) if "dram__bytes_read.sum" in d else 0.0
    kname = v[h.index("Kernel Name")][:34] if "Kernel Name" in h else ""
    stalls = sorted(((val, n[len("smsp__average_warps_issue_stalled_"):-len("_per_issue_active.ratio")])
                     for n, (val, _) in d.items()
                     if n.startswith("smsp__average_warps_issue_stalled_") and val is not None), reverse=True)
    extra = []
    for key, lab in [("smsp__warps_active.avg.per_cycle_active", "warps/sch"),
                     ("smsp__warps_eligible.avg.per_cycle_active", "elig/sch"),
                     ("launch__registers_per_thread", "regs"), ("launch__grid_size", "grid"),
                     ("launch__block_size", "blk")]:
        if key in d and d[key][0] is not None:
            extra.append(f"{lab}={d[key][0]:g}")
    name = rep.split("/")[-1].replace(".ncu-rep", "")
    print(f"{name:24s} {kname:34s} {t:8.2f} us  {mb:8.2f} MB  {mb / t * 1e-3 if t else 0:5.2f} TB/s  " + " ".join(extra))
    print(f"{'':59s} stalls/issue: " + "  ".join(f"{n}={x:.1f}" for x, n in stalls[:5]))


def main():
    for rep in sys.argv[1:]:
        out = subprocess.run(["ncu", "--import", rep, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
        rows = list(csv.reader(io.StringIO(out)))
        if len(rows) < 3:
            print(f"{rep}: no kernels")
            continue
        for v in rows[2:]:
            report(rep, rows[0], rows[1], v)


if __name__ == "__main__":
    main()
