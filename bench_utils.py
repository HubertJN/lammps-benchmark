import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

RUNS_DIR = Path("runs")
LOG_DIR = RUNS_DIR / "logs"
OUT_JSON = RUNS_DIR / "benchmark_summary.json"


def sh(cmd: list[str]) -> str:
    """Run a command and return stdout (best-effort)."""
    try:
        r = subprocess.run(cmd, check=False, text=True, capture_output=True)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def get_machine_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "cpu": {},
        "memory": {},
        "gpu": {},
    }

    lscpu = sh(["lscpu"])
    if lscpu:
        def grab(key: str) -> Optional[str]:
            m = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", lscpu, re.M)
            return m.group(1).strip() if m else None

        info["cpu"] = {
            "model_name": grab("Model name"),
            "architecture": grab("Architecture"),
            "sockets": grab("Socket(s)"),
            "cores_per_socket": grab("Core(s) per socket"),
            "threads_per_core": grab("Thread(s) per core"),
            "cpus": grab("CPU(s)"),
        }

    meminfo = sh(["bash", "-lc", "grep -E 'MemTotal|MemFree|MemAvailable' /proc/meminfo"])
    if meminfo:
        mem = {}
        for line in meminfo.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                key = parts[0].rstrip(":")          
                value = int(parts[1])               
                unit = parts[2]                     # kB
                mem[key] = {"value": value, "unit": unit}
        info["memory"] = mem

    nvsmi = sh(["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true"])
    if nvsmi:
        info["gpu"]["nvidia_smi_L"] = nvsmi
        info["gpu"]["nvidia_smi_query"] = sh([
            "bash", "-lc",
            "nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total --format=csv,noheader || true"
        ])

    return info


def parse_log(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lammps_version": None,
        "omp_threads": None,
        "mpi_tasks": None,

        "atoms": None,
        "steps": None,
        "loop_time_s": None,
        "cpu_use_pct": None,

        "performance": {  # may be partially missing depending on run
            "tau_per_day": None,
            "timesteps_per_s": None,
            "katom_steps_per_s": None,
        },

        "kspace": {
            "style": None,        # e.g. pppm/dipole
            "accuracy": None,     # e.g. 1.0e-4
            "pppm_grid": None,    # e.g. "54 54 54"
            "pppm_order": None,   # e.g. 5
            "pppm_est_abs_rms_force_acc": None,
            "pppm_est_rel_force_acc": None,
            "fft_backend": None,  # e.g. FFTW3
        },

        "timing_breakdown_pct": {},  # Pair/Kspace/Neigh/Comm/Output/Modify/Other
        "neighbors": {
            "total_neighbors": None,
            "ave_neighs_per_atom": None,
            "neighbor_list_builds": None,
            "dangerous_builds": None,
        },

        "total_wall_time": None,  # e.g. 0:00:30
        "warnings": [],
        "errors": [],
    }

    # LAMMPS version
    m = re.search(r"^LAMMPS\s+\((.+)\)\s*$", text, re.M)
    if m:
        out["lammps_version"] = m.group(1).strip()

    # MPI x OMP line
    m = re.search(r"using\s+(\d+)\s+OpenMP thread\(s\)\s+per MPI task", text)
    if m:
        out["omp_threads"] = int(m.group(1))
    m = re.search(r"CPU use with\s+(\d+)\s+MPI tasks?\s*x\s*(\d+)\s+OpenMP", text)
    if m:
        out["mpi_tasks"] = int(m.group(1))
        out["omp_threads"] = int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)%\s+CPU use", text)
    if m:
        out["cpu_use_pct"] = float(m.group(1))

    # Loop time line: Loop time of X on N procs for S steps with A atoms
    m = re.search(r"Loop time of\s+([0-9.]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps\s+with\s+(\d+)\s+atoms", text)
    if m:
        out["loop_time_s"] = float(m.group(1))
        out["steps"] = int(m.group(3))
        out["atoms"] = int(m.group(4))

    # Performance line: Performance: 4680.949 tau/day, 27.089 timesteps/s, 81.266 katom-step/s
    for line in text.splitlines():
        if "Performance:" in line:
            m_tau = re.search(r"([0-9.]+)\s*tau/day", line)
            m_tps = re.search(r"([0-9.]+)\s*timesteps/s", line)
            m_kat = re.search(r"([0-9.]+)\s*katom-step/s", line)
            if m_tau:
                out["performance"]["tau_per_day"] = float(m_tau.group(1))
            if m_tps:
                out["performance"]["timesteps_per_s"] = float(m_tps.group(1))
            if m_kat:
                out["performance"]["katom_steps_per_s"] = float(m_kat.group(1))

    # kspace_style (already fine)
    m = re.search(r"^\s*kspace_style\s+(\S+)\s+([0-9.eE+-]+)(?:\s*#.*)?\s*$", text, re.M)
    if m:
        out["kspace"]["style"] = m.group(1)
        out["kspace"]["accuracy"] = m.group(2)

    # PPPM details: match the actual lines shown in your output (take last occurrence)
    m = None
    for m in re.finditer(r"^\s*grid\s*=\s*([0-9]+\s+[0-9]+\s+[0-9]+)\s*$", text, re.M):
        pass
    if m:
        out["kspace"]["pppm_grid"] = m.group(1)

    m = None
    for m in re.finditer(r"^\s*stencil\s+order\s*=\s*([0-9]+)\s*$", text, re.M):
        pass
    if m:
        out["kspace"]["pppm_order"] = int(m.group(1))

    m = None
    for m in re.finditer(r"^\s*estimated\s+absolute\s+RMS\s+force\s+accuracy\s*=\s*([0-9.eE+-]+)\s*$", text, re.M):
        pass
    if m:
        out["kspace"]["pppm_est_abs_rms_force_acc"] = m.group(1)

    m = None
    for m in re.finditer(r"^\s*estimated\s+relative\s+force\s+accuracy\s*=\s*([0-9.eE+-]+)\s*$", text, re.M):
        pass
    if m:
        out["kspace"]["pppm_est_rel_force_acc"] = m.group(1)

    m = None
    for m in re.finditer(r"^\s*using\s+(?:single|double)\s+precision\s+(\S+)\s*$", text, re.M):
        pass
    if m:
        out["kspace"]["fft_backend"] = m.group(1)

    # Timing breakdown table: grab %total column for sections
    # Example row: Kspace  | 2.5266 | ... | 68.44
    timing_section_re = re.compile(r"^(Pair|Kspace|Neigh|Comm|Output|Modify|Other)\s*\|.*\|\s*([0-9.]+)\s*$", re.M)
    for sec, pct in timing_section_re.findall(text):
        out["timing_breakdown_pct"][sec] = float(pct)

    # Neighbor stats
    m = re.search(r"Total # of neighbors\s*=\s*([0-9]+)", text)
    if m:
        out["neighbors"]["total_neighbors"] = int(m.group(1))
    m = re.search(r"Ave neighs/atom\s*=\s*([0-9.]+)", text)
    if m:
        out["neighbors"]["ave_neighs_per_atom"] = float(m.group(1))
    m = re.search(r"Neighbor list builds\s*=\s*([0-9]+)", text)
    if m:
        out["neighbors"]["neighbor_list_builds"] = int(m.group(1))
    m = re.search(r"Dangerous builds\s*=\s*([0-9]+)", text)
    if m:
        out["neighbors"]["dangerous_builds"] = int(m.group(1))
    m = re.search(r"Dangerous builds not checked", text)
    if m and out["neighbors"]["dangerous_builds"] is None:
        out["neighbors"]["dangerous_builds"] = "not_checked"

    # Total wall time
    m = re.search(r"Total wall time:\s*(.+)$", text, re.M)
    if m:
        out["total_wall_time"] = m.group(1).strip()

    # Warnings / errors (for audit trail)
    out["warnings"] = re.findall(r"^WARNING:\s*(.+)$", text, re.M)
    out["errors"] = re.findall(r"^ERROR:\s*(.+)$", text, re.M)

    # Convenience: identify slowest section
    if out["timing_breakdown_pct"]:
        sec, pct = max(out["timing_breakdown_pct"].items(), key=lambda kv: kv[1])
        out["slowest_section"] = sec
        out["slowest_pct"] = pct
    else:
        out["slowest_section"] = None
        out["slowest_pct"] = None

    return out

def collect_logs_to_json(log_dir: Path, out_json: Path, *, runs_dir: Path = RUNS_DIR) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "machine": get_machine_info(),
        "runs": [],
    }

    runs_by_tag: Dict[str, Dict[str, Any]] = {}

    for log_path in sorted(log_dir.glob("*.log")):
        tag = log_path.stem
        text = log_path.read_text(errors="replace")
        run_data = parse_log(text)
        runs_by_tag[tag] = {"tag": tag, "log_path": str(log_path), **run_data}

    # Merge runner metadata (e.g., timeouts/return codes) from runs/<tag>/run_result.json.
    for result_path in sorted(runs_dir.glob("*/run_result.json")):
        tag = result_path.parent.name
        runner = read_json(result_path)

        run = runs_by_tag.get(tag)
        if run is None:
            # Ensure the run appears in the JSON summary even if no log was captured.
            run = {"tag": tag, "log_path": None, **parse_log("")}

        run["runner"] = runner
        runs_by_tag[tag] = run

    summary["runs"] = [runs_by_tag[k] for k in sorted(runs_by_tag.keys())]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")
    return summary

def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}