import argparse
import math
import os
from itertools import product
from multiprocessing import Pool
from pathlib import Path

from bench_utils import (
    canonical_params,
    collect_logs_to_json,
    index_existing_runs,
    normalize_value,
    read_json,
    reserve_run_ids,
    run_complete,
    run_lammps_job,
)


DEFAULT_LMP = "mylammps/build/lmp"
DEFAULT_INPUT = "in.performance_test.lmp"
DEFAULT_MANUAL_INPUT = "in.manual.lmp"
DEFAULT_MANUAL_TAG = "manual"
DEFAULT_MAX_PARALLEL = "4"
DEFAULT_TIMEOUT_PADDING_S =  "30"


DEFAULT_SWEEP = {
    "ks": [
        "ewald_dipole",
        "pppm_dipole",
    ],
    "kacc": ["1.0e-3", "1.0e-4", "1.0e-5", "1.0e-6"],
    "dcut": [4, 5, 6, 7, 8, 9, 10],
}


def parse_csv(s: str) -> list[str]:
    return [p.strip() for p in str(s).split(",") if p.strip()]


def parse_csv_ints(s: str) -> list[int]:
    return [int(p) for p in parse_csv(s)]


def _normalize_sweep_params(params: dict, sweep_keys: list[str]) -> None:
    for k in sweep_keys:
        if k in params:
            params[k] = normalize_value(k, params[k])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run LAMMPS benchmark sweep, collect logs, and write benchmark_summary.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--runs-dir", default="runs", help="Directory containing run_* folders")
    ap.add_argument("--lmp", default=DEFAULT_LMP, help="Path to LAMMPS executable")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="LAMMPS input script for sweep runs")
    ap.add_argument("--manual-input", default=DEFAULT_MANUAL_INPUT, help="LAMMPS input script for manual baseline")
    ap.add_argument("--manual-tag", default=DEFAULT_MANUAL_TAG, help="Run directory tag for manual baseline")
    ap.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL, help="Max parallel LAMMPS runs")
    ap.add_argument("--timeout-padding-s", type=float, default=DEFAULT_TIMEOUT_PADDING_S, help="Extra seconds added to derived timeout from manual runtime")
    ap.add_argument("--ks", default=",".join(DEFAULT_SWEEP["ks"]), help="Comma-separated kspace styles")
    ap.add_argument("--kacc", default=",".join(DEFAULT_SWEEP["kacc"]), help="Comma-separated kspace accuracies")
    ap.add_argument("--dcut", default=",".join(str(x) for x in DEFAULT_SWEEP["dcut"]), help="Comma-separated dcut values")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    lmp = str(args.lmp)
    inp = str(args.input)
    manual_inp = str(args.manual_input)
    manual_tag = str(args.manual_tag)

    max_parallel = int(args.max_parallel)
    timeout_padding_s = float(args.timeout_padding_s)

    sweep = {
        "ks": parse_csv(args.ks),
        "kacc": parse_csv(args.kacc),
        "dcut": parse_csv_ints(args.dcut),
    }

    sweep_keys = [k for k in sweep.keys() if k != "ks"]
    sweep_values = [sweep[k] for k in sweep_keys]

    include_sweep = ["input", "lmp", "ks"] + sweep_keys
    existing_sweep = index_existing_runs(
        runs_dir,
        run_glob="run_*",
        include_keys=include_sweep,
        normalize=lambda p: _normalize_sweep_params(p, sweep_keys),
    )

    pending: list[tuple[str | None, str, dict]] = []
    skipped: list[tuple[str, str, dict]] = []

    for ks in sweep["ks"]:
        for combo in product(*sweep_values):
            combo_dict = {k: normalize_value(k, v) for k, v in zip(sweep_keys, combo)}
            desired_params = {"input": inp, "lmp": lmp, "ks": ks, **combo_dict}
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

    need_new = sum(1 for rid, _, _ in pending if rid is None)
    new_ids = reserve_run_ids(runs_dir, need_new)
    new_it = iter(new_ids)
    jobs = [(rid if rid is not None else next(new_it), ks, combo_dict) for rid, ks, combo_dict in pending]

    manual_dir = runs_dir / manual_tag
    manual_params = {"run_id": manual_tag, "case": "manual", "input": manual_inp, "lmp": lmp}
    include_manual = ["case", "input", "lmp"]
    manual_sig = canonical_params(manual_params, include_keys=include_manual)
    existing_manual = index_existing_runs(runs_dir, run_glob=manual_tag, include_keys=include_manual).get(manual_sig)
    manual_time_s = None

    if existing_manual == manual_tag and run_complete(manual_dir):
        recorded = read_json(manual_dir / "run_result.json")
        recorded_time = recorded.get("time_s")
        if isinstance(recorded_time, (int, float)) and recorded_time > 0:
            manual_time_s = float(recorded_time)
            print(
                f"SKIP: {manual_tag} existing params match input={manual_inp} "
                f"(using recorded time={manual_time_s:.2f}s)"
            )
        else:
            print(
                f"INFO: {manual_tag} exists but has no usable recorded runtime; "
                f"re-running to derive timeout"
            )

    if manual_time_s is None:
        manual_res = run_lammps_job(
            {
                "run_id": manual_tag,
                "run_dir": str(manual_dir),
                "lmp": lmp,
                "input": manual_inp,
                "tag": manual_tag,
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
        print(f"DONE: {manual_res['run_id']} {manual_status} time={manual_res['time_s']:.2f}s input={manual_inp}")
        if manual_res["returncode"] == 0 and not manual_res.get("timed_out"):
            manual_time_s = float(manual_res["time_s"])

    rounded_manual_s = math.ceil(float(manual_time_s) / 60.0) * 60.0
    sweep_timeout_s = rounded_manual_s + timeout_padding_s
    print(
        f"INFO: sweep timeout set from manual runtime: "
        f"ceil({manual_time_s:.2f}s to nearest minute)={rounded_manual_s:.2f}s "
        f"+ {timeout_padding_s:.0f}s = {sweep_timeout_s:.2f}s"
    )

    for run_id, ks, combo_dict in skipped:
        print(
            f"SKIP: {run_id} existing params match "
            f"ks={ks} "
            + " ".join(f"{k}={combo_dict[k]}" for k in sweep_keys)
        )

    sweep_jobs: list[dict] = []
    for run_id, ks, combo_dict in jobs:
        run_dir = runs_dir / run_id
        params = {"run_id": run_id, "ks": ks, "input": inp, "lmp": lmp, **combo_dict}
        vars_dict = {"ks": ks, **{k: combo_dict[k] for k in sweep_keys}}
        sweep_jobs.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "lmp": lmp,
                "input": inp,
                "tag": run_id,
                "vars": vars_dict,
                "params": params,
                "meta": {"ks": ks, **combo_dict},
                "suppress_output": True,
                "timeout_s": sweep_timeout_s,
            }
        )

    if sweep_jobs:
        n_procs = min(max_parallel, len(sweep_jobs))
        with Pool(processes=n_procs) as pool:
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

    log_dir = runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for run_log in runs_dir.glob("run_*/lammps.log"):
        (log_dir / f"{run_log.parent.name}.log").write_text(run_log.read_text(errors="replace"))

    manual_log = runs_dir / manual_tag / "lammps.log"
    if manual_log.exists():
        (log_dir / f"{manual_tag}.log").write_text(manual_log.read_text(errors="replace"))

    out_json = runs_dir / "benchmark_summary.json"
    summary = collect_logs_to_json(log_dir, out_json, runs_dir=runs_dir)
    print(f"WROTE: {out_json}  runs={len(summary['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
