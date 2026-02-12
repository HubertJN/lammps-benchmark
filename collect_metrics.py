import math
import os
from itertools import product
from multiprocessing import Pool
from pathlib import Path

from bench_utils import *

LMP = "mylammps/build/lmp"
INP = "in.performance_test.lmp"
MANUAL_INP = "in.manual.lmp"
MANUAL_TAG = "manual"

MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "8"))
TIMEOUT_PADDING_S = float(os.environ.get("TIMEOUT_PADDING_S", "300"))  # add 5 minutes by default
if TIMEOUT_PADDING_S < 0:
    TIMEOUT_PADDING_S = 0.0

FALLBACK_TIMEOUT_S = float(os.environ.get("RUN_TIMEOUT_S", "1800"))  # used only if manual timing unavailable
if FALLBACK_TIMEOUT_S <= 0:
    FALLBACK_TIMEOUT_S = None

# Sweep over kspace styles directly.
SWEEP = {
    "ks": [
        "ewald_dipole",
        "pppm_dipole",
        "ewald_disp",
    ],
    "kacc": ["1.0e-3", "1.0e-4", "1.0e-5", "1.0e-6"],
    "dcut": [4, 5, 6, 7, 8, 9, 10],
}

runs_dir = Path("runs")
runs_dir.mkdir(parents=True, exist_ok=True)

sweep_keys = [k for k in SWEEP.keys() if k != "ks"]
sweep_values = [SWEEP[k] for k in sweep_keys]


# Index existing sweep runs once in the parent process (normalize sweep values for stable matching)
include_sweep = ["input", "lmp", "ks"] + sweep_keys
existing_sweep = index_existing_runs(
    runs_dir,
    run_glob="run_*",
    include_keys=include_sweep,
    normalize=lambda p: [p.__setitem__(k, normalize_value(k, p[k])) for k in sweep_keys if k in p],
)


# Build pending sweep cases (skip if matching params + non-empty log exists)
pending = []  # list[(run_id_or_None, ks, combo_dict)]
skipped = []  # list[(run_id, ks, combo_dict)]

for ks in SWEEP["ks"]:
    for combo in product(*sweep_values):
        combo_dict = {k: normalize_value(k, v) for k, v in zip(sweep_keys, combo)}
        desired_params = {"input": INP, "lmp": LMP, "ks": ks, **combo_dict}
        sig = canonical_params(desired_params, include_keys=include_sweep)

        existing_id = existing_sweep.get(sig)
        if existing_id is not None:
            run_dir = runs_dir / existing_id
            if run_complete(run_dir):
                skipped.append((existing_id, ks, combo_dict))
                continue
            pending.append((existing_id, ks, combo_dict))
            continue

        pending.append((None, ks, combo_dict))


# Allocate new run IDs only for truly-missing cases
need_new = sum(1 for rid, _, _ in pending if rid is None)
new_ids = reserve_run_ids(runs_dir, need_new)
new_it = iter(new_ids)
jobs = [(rid if rid is not None else next(new_it), ks, combo_dict) for rid, ks, combo_dict in pending]


if __name__ == "__main__":
    # Manual baseline: skip if params match + log exists
    manual_dir = runs_dir / MANUAL_TAG
    manual_params = {"run_id": MANUAL_TAG, "case": "manual", "input": MANUAL_INP, "lmp": LMP}
    include_manual = ["case", "input", "lmp"]
    manual_sig = canonical_params(manual_params, include_keys=include_manual)
    existing_manual = index_existing_runs(runs_dir, run_glob=MANUAL_TAG, include_keys=include_manual).get(manual_sig)
    manual_time_s = None

    if existing_manual == MANUAL_TAG and run_complete(manual_dir):
        recorded = read_json(manual_dir / "run_result.json")
        recorded_time = recorded.get("time_s")
        if isinstance(recorded_time, (int, float)) and recorded_time > 0:
            manual_time_s = float(recorded_time)
            print(
                f"SKIP: {MANUAL_TAG} existing params match input={MANUAL_INP} "
                f"(using recorded time={manual_time_s:.2f}s)"
            )
        else:
            print(
                f"INFO: {MANUAL_TAG} exists but has no usable recorded runtime; "
                f"re-running to derive timeout"
            )

    if manual_time_s is None:
        manual_res = run_lammps_job(
            {
                "run_id": MANUAL_TAG,
                "run_dir": str(manual_dir),
                "lmp": LMP,
                "input": MANUAL_INP,
                "tag": MANUAL_TAG,
                "vars": {},
                "params": manual_params,
                "meta": {},
                "suppress_output": True,
                "timeout_s": None,
            }
        )
        if manual_res.get("timed_out"):
            manual_status = f"TIMEOUT({manual_res.get('timeout_s')}s)"
        else:
            manual_status = "OK" if manual_res["returncode"] == 0 else f"FAIL(rc={manual_res['returncode']})"
        print(f"DONE: {manual_res['run_id']} {manual_status} time={manual_res['time_s']:.2f}s input={MANUAL_INP}")
        if manual_res["returncode"] == 0 and not manual_res.get("timed_out"):
            manual_time_s = float(manual_res["time_s"])

    if manual_time_s is not None:
        rounded_manual_s = math.ceil(float(manual_time_s) / 60.0) * 60.0
        sweep_timeout_s = rounded_manual_s + TIMEOUT_PADDING_S
        print(
            f"INFO: sweep timeout set from manual runtime: "
            f"ceil({manual_time_s:.2f}s to nearest minute)={rounded_manual_s:.2f}s "
            f"+ {TIMEOUT_PADDING_S:.0f}s = {sweep_timeout_s:.2f}s"
        )
    else:
        sweep_timeout_s = FALLBACK_TIMEOUT_S
        print(
            f"WARNING: could not derive timeout from manual run; "
            f"using fallback RUN_TIMEOUT_S={sweep_timeout_s}"
        )

    for run_id, ks, combo_dict in skipped:
        print(
            f"SKIP: {run_id} existing params match "
            f"ks={ks} "
            + " ".join(f"{k}={combo_dict[k]}" for k in sweep_keys)
        )

    sweep_jobs = []
    for run_id, ks, combo_dict in jobs:
        run_dir = runs_dir / run_id
        params = {"run_id": run_id, "ks": ks, "input": INP, "lmp": LMP, **combo_dict}
        vars_dict = {"ks": ks, **{k: combo_dict[k] for k in sweep_keys}}
        sweep_jobs.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "lmp": LMP,
                "input": INP,
                "tag": run_id,
                "vars": vars_dict,
                "params": params,
                "meta": {"ks": ks, **combo_dict},
                "suppress_output": True,
                "timeout_s": sweep_timeout_s,
            }
        )

    if sweep_jobs:
        with Pool(processes=MAX_PARALLEL) as pool:
            for res in pool.imap_unordered(run_lammps_job, sweep_jobs):
                if res.get("timed_out"):
                    status = f"TIMEOUT({float(sweep_timeout_s):.2f}s)"
                else:
                    status = "OK" if res["returncode"] == 0 else f"FAIL(rc={res['returncode']})"
                print(
                    f"DONE: {res['run_id']} {status} time={res['time_s']:.2f}s "
                    f"ks={res['ks']} "
                    + " ".join(f"{k}={res[k]}" for k in sweep_keys)
                )

    # Build benchmark_summary.json from all run logs
    log_dir = runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for run_log in runs_dir.glob("run_*/lammps.log"):
        (log_dir / f"{run_log.parent.name}.log").write_text(run_log.read_text(errors="replace"))

    manual_log = runs_dir / MANUAL_TAG / "lammps.log"
    if manual_log.exists():
        (log_dir / f"{MANUAL_TAG}.log").write_text(manual_log.read_text(errors="replace"))

    out_json = runs_dir / "benchmark_summary.json"
    summary = collect_logs_to_json(log_dir, out_json)
    print(f"WROTE: {out_json}  runs={len(summary['runs'])}")
