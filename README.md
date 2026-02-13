
# LAMMPS benchmark, parameter sweep and report generator

This is a personal project that started as me tuning an input script (for myself and for collaborator workflows), and then grew into a full automation suite: run a manual baseline, sweep key simulation parameters, collect parsed metrics, and generate a final PDF performance report.

## What this project does

- Runs a **manual baseline** (`in.manual.lmp`)
- Runs an automated **parameter sweep** (`in.performance_test.lmp`) across:
  - `ks` (kspace style)
  - `kacc` (kspace accuracy)
  - `dcut` (dipole cutoff)
- Parses logs and writes a consolidated summary JSON
- Builds a PDF report with ranked runs, speedups, timing pies, and timeout table
- Generates Slurm **scaling run scripts** (and optionally submits them), writing a `scaling_summary.json`

Main scripts:

- `collect_metrics.py` -> executes manual + sweep runs and writes `runs/benchmark_summary.json`
- `plot_metrics.py` -> generates `runs/performance_review.pdf`
- `scaling_analysis.py` -> generates (and optionally submits) scaling `job.slurm` scripts under a chosen output directory

---

## Requirements

### System / build tools

- Linux
- Python 3.10+ (with `venv` and `pip`)
- CMake
- C++ compiler toolchain (e.g. `gcc/g++`)
- MPI (recommended for scaling runs; optional for single-rank runs)
- FFTW3 (recommended; used for KSPACE and Kokkos FFT)

> Note: A GPU is **not** required for this project. The build instructions below produce a **CPU-only** Kokkos (OpenMP) LAMMPS binary.

### LAMMPS packages/features required

Your `lmp` binary should be built with:

- `KOKKOS` (OpenMP backend)
- `OPENMP`
- `DIPOLE`
- `KSPACE`

Recommended:
- `FFTW3` (for `FFT=FFTW3` and `FFT_KOKKOS=FFTW3`)

### Python packages required

- `reportlab`

Install with:

```bash
python3 -m pip install reportlab
````

---

## Installation

### 1) Clone this repository

```bash
git clone git@github.com:HubertJN/lammps-benchmark.git
cd lammps-benchmark
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install reportlab
```

### 3) Build LAMMPS (CPU-only Kokkos + FFTW3)

This repo expects the LAMMPS source tree to live at:

`mylammps/`

and the built executable at:

`mylammps/build/lmp`

#### Download LAMMPS into `mylammps/`

From the repo root:

```bash
git clone -b release https://github.com/lammps/lammps.git mylammps
```

#### Configure + build

From the repo root:

```bash
cmake -S mylammps/cmake -B mylammps/build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D PKG_DIPOLE=ON \
  -D PKG_KSPACE=ON \
  -D FFT=FFTW3 \
  -D FFT_KOKKOS=FFTW3 \
  -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -D CMAKE_INSTALL_RPATH='$ORIGIN' \
  -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
```

Compile:

```bash
cmake --build mylammps/build --parallel
```

> If your build node is memory-constrained or the link step is slow, limit parallelism:
> `cmake --build mylammps/build --parallel 4`

If your binary is somewhere else, pass `--lmp /path/to/lmp` to `collect_metrics.py`.

### 4) Verify the binary has required packages

```bash
bash verify.sh
```

### 5) Set runtime controls

```
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export FFTW_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

---

## How to run everything

Tip: see all options with:

```bash
python collect_metrics.py --help
python scaling_analysis.py --help
```

From the repo root and within the virtual environment:

```bash
python collect_metrics.py
python plot_metrics.py
```

This will:

1. Run manual baseline (`runs/manual`)
2. Run sweep cases (`runs/run_*`)
3. Parse logs into `runs/benchmark_summary.json`
4. Generate report at `runs/performance_review.pdf`

---

## Outputs

* `runs/manual/` -> manual baseline run artifacts
* `runs/run_*/` -> sweep run artifacts (`params.json`, `lammps.log`, `run_result.json`)
* `runs/logs/*.log` -> consolidated log copies
* `runs/benchmark_summary.json` -> parsed machine + run metrics
* `runs/performance_review.pdf` -> final benchmark report

Scaling (Slurm script generation):

* `runs/scaling_*/` -> generated scaling jobs (`job.slurm`, `params.json`)
* `runs/scaling_*/scaling_summary.json` -> machine + slurm config + list of generated scaling jobs
* `runs/scaling_*/submit_result.json` -> `sbatch` output (only when `--submit`)

---

## Scaling runs (Slurm)

1. Edit `slurm_config.yaml` for your cluster (`ACCOUNT`, `CORES_PER_NODE`, `PARTITION`, `TIME_LIMIT`).

2. Generate scripts (no submit):

```bash
python scaling_analysis.py --runs-dir runs/scaling
```

3. Generate + submit:

```bash
python scaling_analysis.py --runs-dir runs/scaling --submit
```

After submission, it will optionally prompt you to monitor jobs using `squeue`.

---

## Notes

* Sweep dimensions are defined in `SWEEP` in `collect_metrics.py`.
* The sweep input script consumes runtime variables (`ks`, `kacc`, `dcut`) via `-var`.
* Existing successful runs are reused/skipped to avoid rerunning identical parameter sets.
