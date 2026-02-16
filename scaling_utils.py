from __future__ import annotations

from pathlib import Path

from itertools import product
from datetime import datetime
import json
import math
import os
import shlex
import shutil
import subprocess
import sys

from bench_utils import get_machine_info


def _strip_inline_comment(s: str) -> str:
    # Removes inline comments of the form: value  # comment
    # Does not attempt to be a full YAML parser (good enough for our simple config).
    if "#" not in s:
        return s
    return s.split("#", 1)[0].rstrip()


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        return s[1:-1]
    return s


def load_slurm_config(path: str | Path) -> dict:
    """Load a simple key: value YAML file without requiring PyYAML.

    Expected keys:
      - ACCOUNT (string)
      - CORES_PER_NODE (int)
      - PARTITION (string)
      - TIME_LIMIT (string, HH:MM:SS)
    """

    text = Path(path).read_text()
    cfg: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_inline_comment(line)
        if not line:
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = _unquote(v.strip())
        if k:
            cfg[k] = v

    missing = [k for k in ("ACCOUNT", "CORES_PER_NODE", "PARTITION", "TIME_LIMIT") if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    return {
        "ACCOUNT": str(cfg["ACCOUNT"]),
        "CORES_PER_NODE": int(cfg["CORES_PER_NODE"]),
        "PARTITION": str(cfg["PARTITION"]),
        "TIME_LIMIT": str(cfg["TIME_LIMIT"]),
    }


def parse_csv(s: str) -> list[str]:
    return [p.strip() for p in str(s).split(",") if p.strip()]


def parse_csv_ints(s: str) -> list[int]:
    return [int(p) for p in parse_csv(s)]


def prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    """Interactive y/n prompt.

    Returns False automatically if stdin is not a TTY.
    """

    if not sys.stdin.isatty():
        return False
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return bool(default)
    return ans in {"y", "yes"}


def monitor_slurm_jobs_interactive() -> None:
    """Ask for username (default $USER) and run `squeue -u <user>`.

    No-ops if stdin is not a TTY or if `squeue` is not available.
    """

    if not sys.stdin.isatty():
        return

    squeue = shutil.which("squeue")
    if not squeue:
        print("INFO: 'squeue' not found in PATH; cannot monitor jobs.")
        return

    default_user = os.environ.get("USER") or ""
    try:
        entered = input(f"Slurm username to monitor (CTRL + C to stop monitoring) [{default_user}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    username = entered or default_user
    if not username:
        print("INFO: No username provided; skipping monitoring.")
        return

    try:
        subprocess.run(["watch", "squeue", "-u", username], check=False)
    except KeyboardInterrupt:
        print("\n--- watch interrupted (Ctrl+C) ---\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _extract_lammps_paths(lammps_cmd: str) -> tuple[str | None, str | None]:
    """Best-effort extraction of lmp binary and -in input from a formatted command."""
    try:
        tokens = shlex.split(lammps_cmd)
    except Exception:
        return None, None

    lmp = tokens[0] if tokens else None
    inp = None
    for i, t in enumerate(tokens):
        if t == "-in" and i + 1 < len(tokens):
            inp = tokens[i + 1]
            break
    return lmp, inp


def generate_slurm_scripts(
    *,
    runs_dir: Path,
    ks_list: list[str],
    kacc_list: list[str],
    dcut_list: list[int],
    cores_list: list[int],
    size_list: list[str],
    cores_per_node: int,
    partition: str,
    time_limit: str,
    account: str,
    lammps_command_template: str,
    slurm_template: str,
    submit: bool,
    mode: str, 
) -> tuple[int, list[tuple[Path, str]]]:
    """Generate run directories containing job.slurm scripts; optionally submit with sbatch."""

    if mode not in {"collect_metrics", "scaling_analysis"}:
        raise ValueError(f"Invalid mode: {mode}")

    runs_dir.mkdir(parents=True, exist_ok=True)

    sbatch = shutil.which("sbatch")
    if submit and not sbatch:
        raise SystemExit("--submit requested but 'sbatch' was not found in PATH")

    n_written = 0
    submitted: list[tuple[Path, str]] = []

    machine = get_machine_info()

    summary: dict = {
        "machine": machine,
        "generator": {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tool": mode,
        },
        "slurm": {
            "partition": partition,
            "time_limit": time_limit,
            "account": account,
            "cores_per_node": int(cores_per_node),
        },
        "runs": [],
    }

    i = 0
    for size in size_list:
        atoms, side = size.split(",")
        atoms = int(atoms.strip())
        side = float(side.strip())
        for ks, kacc, dcut, total_cores in product(ks_list, kacc_list, dcut_list, cores_list):
            total_cores = int(total_cores)
            if mode == "collect_metrics":
                tag = f"run_{i:06d}"
                i += 1
            elif mode == "scaling_analysis":
                tag = f"scaling_{atoms:06d}_{total_cores:03d}"
            run_dir = runs_dir / tag
            log_path = run_dir / "lammps.log"

            run_dir.mkdir(parents=True, exist_ok=True)

            nodes = max(1, math.ceil(total_cores / int(cores_per_node)))
            ntasks_per_node = max(1, math.ceil(total_cores / nodes))

            lammps_cmd = lammps_command_template.format(
                ks=ks,
                kacc=kacc,
                dcut=dcut,
                tag=tag,
                atoms=str(int(int(atoms)/5)),  # LAMMPS input script will replicate to get desired total atoms
                side=side,
                log_path=log_path,
            )

            lmp_path, inp_path = _extract_lammps_paths(lammps_cmd)

            slurm_script = slurm_template.format(
                run_dir=str(run_dir),
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                partition=partition,
                time_limit=time_limit,
                account=account,
                log_dir=str(run_dir),
                lammps_cmd=lammps_cmd,
            )

            slurm_script_path = run_dir / "job.slurm"
            slurm_script_path.write_text(slurm_script)
            n_written += 1

            params = {
                "run_id": tag,
                "case": "scaling",
                "ks": ks,
                "kacc": kacc,
                "dcut": dcut,
                "atoms": atoms,
                "side": side,
                "cores": total_cores,
                "nodes": nodes,
                "ntasks_per_node": ntasks_per_node,
                "lmp": lmp_path,
                "input": inp_path,
                "log_path": str(log_path),
                "slurm": {
                    "partition": partition,
                    "time_limit": time_limit,
                    "account": account,
                    "cores_per_node": int(cores_per_node),
                },
                "generator": {
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "tool": "scaling_analysis",
                },
                "machine": machine,
            }
            _write_json(run_dir / "params.json", params)

            summary["runs"].append(
                {
                    "tag": tag,
                    "run_dir": str(run_dir),
                    "job_slurm": str(slurm_script_path),
                    "params_json": str(run_dir / "params.json"),
                    "log_path": str(log_path),
                    "ks": ks,
                    "kacc": kacc,
                    "dcut": dcut,
                    "atoms": atoms,
                    "side": side,
                    "cores": total_cores,
                    "nodes": nodes,
                    "ntasks_per_node": ntasks_per_node,
                }
            )

            if submit:
                r = subprocess.run(
                    [sbatch, str(slurm_script_path)],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                out = (r.stdout or r.stderr).strip()
                submitted.append((slurm_script_path, out))
                submit_result = {
                    "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "cmd": [sbatch, str(slurm_script_path)],
                    "returncode": int(r.returncode),
                    "stdout": (r.stdout or ""),
                    "stderr": (r.stderr or ""),
                    "note": out,
                }
                _write_json(run_dir / "submit_result.json", submit_result)

                summary["runs"][-1]["submitted"] = True
                summary["runs"][-1]["submit_result_json"] = str(run_dir / "submit_result.json")
                summary["runs"][-1]["sbatch_note"] = out
            else:
                summary["runs"][-1]["submitted"] = False

    # One-file summary of all generated jobs under runs_dir.
    if mode == "collect_metrics":
        summary_path = runs_dir / "metrics_slurm_summary.json"
    elif mode == "scaling_analysis":
        summary_path = runs_dir / "scaling_slurm_summary.json"
    _write_json(summary_path, summary)

    return n_written, submitted