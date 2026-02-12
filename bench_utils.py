import json
import os
import platform
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Iterable, Tuple

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

    # Kspace style & accuracy (captures the executed line like: kspace_style pppm/dipole  1.0e-4)
    m = re.search(r"^\s*kspace_style\s+(\S+)\s+([0-9.eE+-]+)\s*$", text, re.M)
    if m:
        out["kspace"]["style"] = m.group(1)
        out["kspace"]["accuracy"] = m.group(2)

    # PPPM details (if present)
    m = re.search(r"grid\s*=\s*([0-9]+\s+[0-9]+\s+[0-9]+)", text)
    if m:
        out["kspace"]["pppm_grid"] = m.group(1)
    m = re.search(r"stencil order\s*=\s*([0-9]+)", text)
    if m:
        out["kspace"]["pppm_order"] = int(m.group(1))
    m = re.search(r"estimated absolute RMS force accuracy\s*=\s*([0-9.eE+-]+)", text)
    if m:
        out["kspace"]["pppm_est_abs_rms_force_acc"] = m.group(1)
    m = re.search(r"estimated relative force accuracy\s*=\s*([0-9.eE+-]+)", text)
    if m:
        out["kspace"]["pppm_est_rel_force_acc"] = m.group(1)
    m = re.search(r"using\s+(?:single|double)\s+precision\s+(\S+)", text)
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

def next_run_id(base: Path) -> str:
    existing = [p for p in base.glob("run_*") if p.is_dir()]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_", 1)[1]))
        except Exception:
            pass
    last = max(nums) if nums else 0
    return f"run_{last + 1:06d}"

def normalize_value(key: str, val) -> str:
    # ensures values are string-safe for -var
    if key == "dcut":
        return f"{float(val):.1f}"
    return str(val)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def canonical_params(params: Dict[str, Any], *, include_keys: list[str]) -> Tuple[Tuple[str, str], ...]:
    """Return a stable signature for params matching (ignores run_id/time/etc)."""
    items: list[tuple[str, str]] = []
    for k in include_keys:
        v = params.get(k)
        items.append((k, "" if v is None else str(v)))
    return tuple(sorted(items))


def run_complete(run_dir: Path, *, log_name: str = "lammps.log") -> bool:
    """Return True only if the run log appears complete/successful.

    This guards against cases where params.json was written but the simulation
    crashed or exited early.
    """
    p = run_dir / log_name
    try:
        if not (p.exists() and p.stat().st_size > 0):
            return False
        text = p.read_text(errors="replace")
    except Exception:
        return False

    parsed = parse_log(text)

    # Any ERROR lines indicate an unsuccessful run.
    if parsed.get("errors"):
        return False

    perf = (parsed.get("performance") or {}).get("timesteps_per_s")
    if perf is None:
        return False

    # Confirm the run reached the usual end-of-run markers.
    if parsed.get("total_wall_time") is None and parsed.get("loop_time_s") is None:
        return False

    return True


def reserve_run_ids(base: Path, n: int, *, prefix: str = "run_") -> list[str]:
    """Allocate run_XXXXXX IDs safely in the parent process (no races)."""
    existing = [p for p in base.glob(f"{prefix}*") if p.is_dir()]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_", 1)[1]))
        except Exception:
            pass
    last = max(nums) if nums else 0
    return [f"{prefix}{last + i:06d}" for i in range(1, n + 1)]


def index_existing_runs(
    base: Path,
    *,
    run_glob: str,
    include_keys: list[str],
    normalize: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[Tuple[Tuple[str, str], ...], str]:
    """Map canonical param signature -> existing run_id (dir name)."""
    sig_to_id: Dict[Tuple[Tuple[str, str], ...], str] = {}
    for run_dir in sorted(base.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        params_path = run_dir / "params.json"
        if not params_path.exists():
            continue
        params = read_json(params_path)
        if normalize is not None:
            try:
                normalize(params)
            except Exception:
                pass
        sig = canonical_params(params, include_keys=include_keys)
        sig_to_id.setdefault(sig, run_dir.name)
    return sig_to_id


def run_lammps_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Generic LAMMPS runner used by sweep/manual scripts (picklable for multiprocessing)."""
    lmp = job["lmp"]
    inp = job["input"]
    run_id = job["run_id"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    log_name = job.get("log_name", "lammps.log")
    log_path = run_dir / log_name

    params = job.get("params") or {}
    if params:
        (run_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")

    vars_dict = job.get("vars") or {}
    tag = job.get("tag", run_id)

    cmd = [lmp, "-in", inp, "-var", "tag", str(tag), "-log", str(log_path)]
    for k, v in vars_dict.items():
        cmd += ["-var", str(k), str(v)]

    suppress = bool(job.get("suppress_output", True))
    timeout_s = job.get("timeout_s")
    if timeout_s is not None:
        try:
            timeout_s = float(timeout_s)
        except Exception:
            timeout_s = None
        if timeout_s is not None and timeout_s <= 0:
            timeout_s = None

    t0 = time.perf_counter()
    timed_out = False
    returncode: int = -1
    note: Optional[str] = None

    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL if suppress else None,
            stderr=subprocess.DEVNULL if suppress else None,
            timeout=timeout_s,
        )
        returncode = int(r.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        note = "timeout"
        returncode = -1
    except Exception as e:
        note = f"exception: {type(e).__name__}"
        returncode = -1

    dt = time.perf_counter() - t0

    meta = job.get("meta") or {}
    result = {
        "run_id": run_id,
        "returncode": returncode,
        "time_s": dt,
        "timed_out": timed_out,
        "timeout_s": timeout_s,
        "note": note,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "cmd": cmd,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **meta,
    }

    # Persist runner metadata so it can be included in benchmark_summary.json.
    try:
        (run_dir / "run_result.json").write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    except Exception:
        pass

    return result