import json
import os
import subprocess
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from bench_utils import *

LMP = "mylammps/build/lmp"
INP = "in.performance_test.lmp"

MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "2"))

cases = {
    "ewald_dipole": "ewald_dipole",
    "pppm_dipole":  "pppm_dipole",
    "ewald_disp":   "ewald_disp",
}

sweeps = {
    "kacc": ["1.0e-4"],
    "dcut": [4, 5],
}

runs_dir = Path("runs")
runs_dir.mkdir(parents=True, exist_ok=True)

sweep_keys = list(sweeps.keys())
sweep_values = [sweeps[k] for k in sweep_keys]

def reserve_run_ids(base: Path, n: int) -> list[str]:
    """Allocate run_XXXXXX IDs safely in the parent process (no races)."""
    existing = [p for p in base.glob("run_*") if p.is_dir()]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_", 1)[1]))
        except Exception:
            pass
    last = max(nums) if nums else 0
    return [f"run_{last + i:06d}" for i in range(1, n + 1)]

# Build job list first (case x sweep)
raw_jobs = []
for case_name, ks in cases.items():
    for combo in product(*sweep_values):
        combo_dict = {k: normalize_value(k, v) for k, v in zip(sweep_keys, combo)}
        raw_jobs.append((case_name, ks, combo_dict))

# Pre-assign run IDs in the parent process (fixes multiprocessing race)
run_ids = reserve_run_ids(runs_dir, len(raw_jobs))
jobs = [(run_ids[i], *raw_jobs[i]) for i in range(len(raw_jobs))]  # (run_id, case_name, ks, combo_dict)

def run_one(job):
    run_id, case_name, ks, combo_dict = job

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "run_id": run_id,
        "case": case_name,
        "ks": ks,
        "input": INP,
        "lmp": LMP,
        **combo_dict,
    }
    (run_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    log_path = run_dir / "lammps.log"
    cmd = [LMP, "-in", INP, "-var", "tag", run_id, "-var", "ks", ks, "-log", str(log_path)]
    for k in sweep_keys:
        cmd += ["-var", k, combo_dict[k]]

    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0

    return {
        "run_id": run_id,
        "case": case_name,
        "ks": ks,
        **combo_dict,
        "returncode": r.returncode,
        "time_s": dt,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
    }

if __name__ == "__main__":
    with Pool(processes=MAX_PARALLEL) as pool:
        for res in pool.imap_unordered(run_one, jobs):
            status = "OK" if res["returncode"] == 0 else f"FAIL(rc={res['returncode']})"
            print(
                f"DONE: {res['run_id']} {status} time={res['time_s']:.2f}s "
                f"case={res['case']} ks={res['ks']} "
                + " ".join(f"{k}={res[k]}" for k in sweep_keys)
            )

    # Build benchmark_summary.json from all run logs
    log_dir = runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for run_log in runs_dir.glob("run_*/lammps.log"):
        (log_dir / f"{run_log.parent.name}.log").write_text(run_log.read_text(errors="replace"))

    out_json = runs_dir / "benchmark_summary.json"
    summary = collect_logs_to_json(log_dir, out_json)
    print(f"WROTE: {out_json}  runs={len(summary['runs'])}")