from __future__ import annotations

import argparse
import time
import json
import math
import re
import shutil
import shlex
import subprocess
from pathlib import Path

from bench_utils import collect_logs_to_json, parse_log, read_json

from scaling_utils import (
    generate_slurm_scripts,
    load_slurm_config,
    monitor_slurm_jobs_interactive,
    parse_csv,
    parse_csv_ints,
    prompt_yes_no,
)

DEFAULT_MANUAL_INPUT = "in.manual.lmp"
DEFAULT_MANUAL_TAG = "manual"

# Match scaling_analysis.py defaults.
PARAMS = {
    "ks": [
        "ewald_dipole",
        "pppm_dipole",
    ],
    "kacc": ["1.0e-3", "1.0e-4", "1.0e-5", "1.0e-6"],
    "dcut": [4, 5, 6, 7, 8, 9, 10],
    "cores" : [1],
}

LAMMPS_COMMAND_TEMPLATE = (
    "mylammps/build/lmp -k on t 1 -sf kk -in in.performance_test.lmp "
    "-var ks {ks} "
    "-var kacc {kacc} "
    "-var dcut {dcut} "
    "-var tag {tag} "
    "-log {log_path}"
)

SLURM_TEMPLATE = """#!/bin/bash

#SBATCH --output={log_dir}/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3850
#SBATCH --partition={partition}
#SBATCH --time={time_limit}

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 IPython FFTW

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export FFTW_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --cpu-bind=cores {lammps_cmd}
"""
def wait_job_afterok(jobid: str, poll_s: int = 10) -> bool:
    sacct = shutil.which("sacct")
    squeue = shutil.which("squeue")
    if not (sacct and squeue):
        raise SystemExit("Need sacct and squeue in PATH to wait for job completion")

    while True:
        # If still in queue, keep waiting
        r = subprocess.run([squeue, "-h", "-j", jobid, "-o", "%T"], text=True, capture_output=True)
        state = (r.stdout or "").strip()
        if state:
            time.sleep(poll_s)
            continue

        # Not in squeue -> finished; ask accounting for final state
        r = subprocess.run([sacct, "-n", "-X", "-j", jobid, "-o", "State"], text=True, capture_output=True)
        states = [s.strip() for s in (r.stdout or "").splitlines() if s.strip()]
        # take first non-empty; sacct may show job + steps
        final = states[0].split()[0] if states else ""
        return final == "COMPLETED"

def _slurm_time_from_seconds(seconds: float) -> str:
    # Slurm expects a walltime; safest is to round up to full minutes.
    s = int(max(0.0, float(seconds)))
    s = max(60, int(math.ceil(s / 60.0) * 60))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _copy_log(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(errors="replace"))


def _parse_walltime_to_seconds(s: str | None) -> float | None:
    if not s:
        return None
    text = str(s).strip()
    # Common formats:
    #   0:00:30
    #   12:34:56
    #   2-12:34:56
    m = re.match(r"^(?:(\d+)-)?(\d+):(\d+):(\d+)$", text)
    if not m:
        return None
    days = int(m.group(1) or 0)
    h = int(m.group(2))
    minutes = int(m.group(3))
    sec = int(m.group(4))
    return float(days * 86400 + h * 3600 + minutes * 60 + sec)


def _latest_slurm_out(run_dir: Path) -> Path | None:
    outs = list(run_dir.glob("slurm-*.out"))
    if not outs:
        return None
    try:
        outs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        outs.sort()
    return outs[0]


def _slurm_out_status(run_dir: Path) -> dict:
    """Best-effort status classification from slurm-*.out."""
    p = _latest_slurm_out(run_dir)
    if p is None:
        return {"state": "missing", "path": None, "note": "no slurm-*.out found"}

    try:
        text = p.read_text(errors="replace")
    except Exception:
        return {"state": "unreadable", "path": str(p), "note": "cannot read slurm output"}

    t = text.lower()

    timeout_markers = [
        "due to time limit",
        "time limit",
        "cancelled",
        "canceled",
    ]
    if any(m in t for m in timeout_markers):
        return {"state": "timeout", "path": str(p), "note": "time limit / cancelled marker in slurm output"}

    error_markers = [
        "srun: error",
        "slurmstepd:",
        "segmentation fault",
        "floating point exception",
        "out of memory",
        "oom-kill",
        "killed",
        "error:",
    ]
    if any(m in t for m in error_markers):
        # Don't misclassify benign "slurmstepd:" lines as errors unless they look like errors.
        if "slurmstepd:" in t and "error" not in t and "failed" not in t and "exceeded" not in t:
            pass
        else:
            return {"state": "error", "path": str(p), "note": "error marker in slurm output"}

    m = re.search(r"exited with exit code\s+(\d+)", t)
    if m:
        try:
            code = int(m.group(1))
        except Exception:
            code = None
        if code not in (None, 0):
            return {"state": "error", "path": str(p), "note": f"exit code {code} in slurm output"}

    return {"state": "ok", "path": str(p), "note": "no timeout/error markers found"}


def _manual_time_seconds(manual_dir: Path) -> float | None:
    st = _slurm_out_status(manual_dir)
    if st.get("state") in {"timeout", "error"}:
        return None
    log_path = manual_dir / "lammps.log"
    if not log_path.exists():
        return None
    parsed = parse_log(log_path.read_text(errors="replace"))
    t = _parse_walltime_to_seconds(parsed.get("total_wall_time"))
    if isinstance(t, (int, float)) and float(t) > 0:
        return float(t)
    lt = parsed.get("loop_time_s")
    if isinstance(lt, (int, float)) and float(lt) > 0:
        return float(lt)
    return None


def _write_manual_slurm(*, run_dir: Path, partition: str, time_limit: str, account: str, lammps_cmd: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    script = SLURM_TEMPLATE.format(
        run_dir=str(run_dir),
        nodes=1,
        ntasks_per_node=1,
        partition=partition,
        time_limit=time_limit,
        account=account,
        log_dir=str(run_dir),
        lammps_cmd=lammps_cmd,
    )
    p = run_dir / "job.slurm"
    p.write_text(script)
    return p


def _submit_sbatch(slurm_path: Path) -> str:
    sbatch = shutil.which("sbatch")
    if not sbatch:
        raise SystemExit("--submit requested but 'sbatch' was not found in PATH")
    # --parsable prints just the jobid (or jobid;cluster)
    r = subprocess.run([sbatch, "--parsable", str(slurm_path)], check=True, text=True, capture_output=True)
    return (r.stdout or "").strip().split(";")[0]


def collect_summary(*, runs_dir: Path, out_json: Path, manual_tag: str) -> dict:
    log_dir = runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Consolidate logs under runs/logs/ (manual + scaling_*).
    _copy_log(runs_dir / manual_tag / "lammps.log", log_dir / f"{manual_tag}.log")
    for p in sorted(runs_dir.glob("scaling_*/lammps.log")):
        _copy_log(p, log_dir / f"{p.parent.name}.log")

    summary = collect_logs_to_json(log_dir, out_json, runs_dir=runs_dir)

    # Enrich with any metadata files present under each run dir.
    for run in summary.get("runs", []):
        tag = run.get("tag")
        if not tag:
            continue
        run_dir = runs_dir / str(tag)
        params_path = run_dir / "params.json"
        if params_path.exists():
            run["params"] = read_json(params_path)
        submit_path = run_dir / "submit_result.json"
        if submit_path.exists():
            run["submit_result"] = read_json(submit_path)

    out_json.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Scaling workflow helper: run manual baseline (to set Slurm walltime), "
            "generate scaling Slurm scripts (optionally submit), or collect metrics into a summary JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", default="slurm_config.yaml", help="Path to slurm_config.yaml")
    ap.add_argument("--runs-dir", default="runs", help="Runs directory")
    ap.add_argument("--lmp", default="mylammps/build/lmp", help="Path to LAMMPS executable")

    ap.add_argument("--manual-input", default=DEFAULT_MANUAL_INPUT, help="Manual baseline input script")
    ap.add_argument("--manual-tag", default=DEFAULT_MANUAL_TAG, help="Run directory name for manual baseline")
    ap.add_argument(
        "--manual",
        action="store_true",
        help="Generate the manual baseline Slurm job (and optionally submit) then exit",
    )
    ap.add_argument("--time-pad-s", type=float, default=0.0, help="Seconds added to manual runtime for Slurm time")

    ap.add_argument("--submit", action="store_true", help="Submit generated scripts with sbatch")
    ap.add_argument("--collect", action="store_true", help="Collect available metrics into a summary JSON and exit")
    ap.add_argument(
        "--out-json",
        default="metrics_summary.json",
        help="Output JSON name (created under runs-dir) for --collect",
    )

    ap.add_argument("--ks", default=",".join(PARAMS["ks"]), help="Comma-separated kspace styles")
    ap.add_argument("--kacc", default=",".join(PARAMS["kacc"]), help="Comma-separated kspace accuracies")
    ap.add_argument("--dcut", default=",".join(str(x) for x in PARAMS["dcut"]), help="Comma-separated dcut values")
    ap.add_argument("--cores", default=",".join(str(x) for x in PARAMS["cores"]), help="Comma-separated core counts")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    manual_tag = str(args.manual_tag)
    manual_dir = runs_dir / manual_tag
    out_json = runs_dir / str(args.out_json)

    if args.collect:
        summary = collect_summary(runs_dir=runs_dir, out_json=out_json, manual_tag=manual_tag)
        print(f"WROTE: {out_json}  runs={len(summary.get('runs', []))}")
        return 0

    cfg = load_slurm_config(args.config)

    manual_out = _slurm_out_status(manual_dir)
    manual_log_time_s = _manual_time_seconds(manual_dir)

    if args.manual:
        # Manual job walltime comes from config; its *measured* runtime is parsed from its log on later runs.
        manual_time_limit = str(cfg["TIME_LIMIT"])
        lammps_cmd = (
            f"{args.lmp} -k on t 1 -sf kk -in {args.manual_input} "
            f"-var tag {manual_tag} -log {manual_dir / 'lammps.log'}"
        )
        slurm_path = _write_manual_slurm(
            run_dir=manual_dir,
            partition=str(cfg["PARTITION"]),
            time_limit=manual_time_limit,
            account=str(cfg["ACCOUNT"]),
            lammps_cmd=lammps_cmd,
        )
        (manual_dir / "params.json").write_text(
            json.dumps(
                {
                    "run_id": manual_tag,
                    "case": "manual",
                    "lmp": str(args.lmp),
                    "input": str(args.manual_input),
                    "slurm": {
                        "partition": str(cfg["PARTITION"]),
                        "time_limit": manual_time_limit,
                        "account": str(cfg["ACCOUNT"]),
                    },
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote manual job: {slurm_path}")
        if args.submit:
            note = _submit_sbatch(slurm_path)
            (manual_dir / "submit_result.json").write_text(
                json.dumps({"note": note, "job_slurm": str(slurm_path)}, indent=2) + "\n"
            )
            print(f"Submitted {slurm_path}: {note}")
            
            print("Waiting for manual job to complete...")
            ok = wait_job_afterok(note)
            if not ok:
                raise SystemExit(f"Manual job {note} did not complete successfully")

    # Require the manual baseline to have been run via --manual and completed.
    if manual_log_time_s is None:
        if manual_out.get("state") in {"timeout", "error"}:
            raise SystemExit(
                f"Manual baseline appears to have {manual_out.get('state')}ed.\n"
                f"Check: {manual_out.get('path')}\n"
                f"Then re-run: python collect_metrics.py --manual --submit"
            )
        raise SystemExit(
            f"Manual baseline runtime not found under: {manual_dir / 'lammps.log'}\n"
            f"Run: python collect_metrics.py --manual --submit  (then wait for it to finish)"
        )

    time_limit = _slurm_time_from_seconds(manual_log_time_s + float(args.time_pad_s or 0.0))
    print(f"INFO: derived Slurm --time={time_limit} from manual runtime")

    # 2) Generate scripts (optionally submit).
    lammps_cmd_template = LAMMPS_COMMAND_TEMPLATE.replace("mylammps/build/lmp", str(args.lmp))

    n_written, submitted = generate_slurm_scripts(
        runs_dir=runs_dir,
        ks_list=parse_csv(args.ks),
        kacc_list=parse_csv(args.kacc),
        dcut_list=parse_csv_ints(args.dcut),
        cores_list=parse_csv_ints(args.cores),
        cores_per_node=int(cfg["CORES_PER_NODE"]),
        partition=str(cfg["PARTITION"]),
        time_limit=time_limit,
        account=str(cfg["ACCOUNT"]),
        lammps_command_template=lammps_cmd_template,
        slurm_template=SLURM_TEMPLATE,
        submit=bool(args.submit),
        mode="collect_metrics",
    )

    print(f"Wrote {n_written} Slurm scripts under: {runs_dir}")
    if args.submit:
        for p, out in submitted:
            print(f"Submitted {p}: {out}")
        if submitted and prompt_yes_no("Monitor Slurm jobs now?", default=False):
            monitor_slurm_jobs_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())