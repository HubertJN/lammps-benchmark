from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_utils import (
    submit_sbatch, 
    wait_job_afterok, 
    manual_time_seconds, 
    slurm_out_status, 
    slurm_time_from_seconds, 
    collect_summary, 
    write_manual_slurm,
)

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

#SBATCH --output={log_dir}/slurm.out
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

    if args.manual:
        # Manual job walltime comes from config; its *measured* runtime is parsed from its log on later runs.
        manual_time_limit = str(cfg["TIME_LIMIT"])
        lammps_cmd = (
            f"{args.lmp} -k on t 1 -sf kk -in {args.manual_input} "
            f"-var tag {manual_tag} -log {manual_dir / 'lammps.log'}"
        )
        slurm_path = write_manual_slurm(
            run_dir=manual_dir,
            partition=str(cfg["PARTITION"]),
            time_limit=manual_time_limit,
            account=str(cfg["ACCOUNT"]),
            lammps_cmd=lammps_cmd,
            SLURM_TEMPLATE=SLURM_TEMPLATE,
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
            note = submit_sbatch(slurm_path)
            (manual_dir / "submit_result.json").write_text(
                json.dumps({"note": note, "job_slurm": str(slurm_path)}, indent=2) + "\n"
            )
            print(f"Submitted {slurm_path}: {note}")
            
            print("Waiting for manual job to complete...")
            ok = wait_job_afterok(note)
            if not ok:
                raise SystemExit(f"Manual job {note} did not complete successfully")
            
    manual_log_time_s = manual_time_seconds(manual_dir)
    manual_out = slurm_out_status(manual_dir)

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

    time_limit = slurm_time_from_seconds(manual_log_time_s + float(args.time_pad_s or 0.0))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())