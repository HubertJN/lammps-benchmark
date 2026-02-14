from pathlib import Path
import argparse

from scaling_utils import (
    load_slurm_config,
    generate_slurm_scripts,
    parse_csv,
    parse_csv_ints,
    prompt_yes_no,
    monitor_slurm_jobs_interactive,
)

# Sweep over kspace styles directly.
PARAMS = {
    "ks": ["pppm_dipole"],
    "kacc": ["1.0e-4"],
    "dcut": [6],
    "cores": [1, 2, 4, 8, 16, 32, 48],
}

LAMMPS_COMMAND_TEMPLATE = (
    "mylammps/build/lmp -k on t 1 -sf kk -in in.scaling_test.lmp "
    "-var ks {ks} "
    "-var kacc {kacc} "
    "-var dcut {dcut} "
    "-var tag {tag} "
    "-log {log_path}"
)

SLURM_TEMPLATE = """#!/bin/bash

#SBATCH --output={log_dir}/slurm.out
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
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
    ap = argparse.ArgumentParser(description="Generate (and optionally submit) Slurm scripts for scaling runs.")
    ap.add_argument("--config", default="slurm_config.yaml", help="Path to slurm_config.yaml")
    ap.add_argument("--runs-dir", default="runs", help="Output directory for generated run folders")
    ap.add_argument("--submit", action="store_true", help="Submit each generated job with sbatch")
    ap.add_argument("--ks", default=",".join(PARAMS["ks"]), help="Comma-separated kspace styles")
    ap.add_argument("--kacc", default=",".join(PARAMS["kacc"]), help="Comma-separated kspace accuracies")
    ap.add_argument("--dcut", default=",".join(str(x) for x in PARAMS["dcut"]), help="Comma-separated dcut values")
    ap.add_argument("--cores", default=",".join(str(x) for x in PARAMS["cores"]), help="Comma-separated core counts")
    args = ap.parse_args(argv)

    cfg = load_slurm_config(args.config)

    n_written, submitted = generate_slurm_scripts(
        runs_dir=Path(args.runs_dir),
        ks_list=parse_csv(args.ks),
        kacc_list=parse_csv(args.kacc),
        dcut_list=parse_csv_ints(args.dcut),
        cores_list=parse_csv_ints(args.cores),
        cores_per_node=int(cfg["CORES_PER_NODE"]),
        partition=str(cfg["PARTITION"]),
        time_limit=str(cfg["TIME_LIMIT"]),
        account=str(cfg["ACCOUNT"]),
        lammps_command_template=LAMMPS_COMMAND_TEMPLATE,
        slurm_template=SLURM_TEMPLATE,
        submit=bool(args.submit),
        mode="scaling_analysis",
    )

    print(f"Wrote {n_written} Slurm scripts under: {args.runs_dir}")
    if args.submit:
        for p, out in submitted:
            print(f"Submitted {p}: {out}")

        if submitted and prompt_yes_no("Monitor Slurm jobs now?", default=False):
            monitor_slurm_jobs_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
