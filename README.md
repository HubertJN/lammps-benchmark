# LAMMPS benchmark, parameter sweep and report generator

This is a personal project that started as me tuning an input script (for myself and for collaborator workflows), and then grew into a full automation suite: run a manual baseline, sweep key simulation parameters, collect parsed metrics, and generate a final PDF performance report. <br>
<b> This project is designed to function on an HPC systems that makes use of SLURM.</b>

## What this project does

- Runs a **manual baseline** (`in.manual.lmp`)
- Runs an automated **parameter sweep** (`in.performance_test.lmp`) across:
  - `ks` (kspace style)
  - `kacc` (kspace accuracy)
  - `dcut` (dipole cutoff)
- Runs an automated <b>core scaling test</b> (`in.scaling_test.lmp`)
- Parses logs and writes a consolidated summary JSON
- Builds a PDF report with ranked runs, speedups, timing pies, timeout and (if available) speed-up data


Main scripts:

- `collect_metrics.py` -> runs the manual baseline to derive a walltime, generates (and optionally submits) scaling `job.slurm` scripts under `runs/`; use `--collect` to write a consolidated metrics JSON
- `plot_metrics.py` -> generates `performance_review.pdf`
- `scaling_analysis.py` -> generates (and optionally submits) scaling `job.slurm` scripts under a chosen output directory

---

## Requirements

### System / build tools

- Linux
- Python 3.10+ (with `venv` and `pip`)
- CMake
- C++ compiler toolchain (e.g. `gcc/g++`)
- MPI
- FFTW3

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

### 5) Slurm config

Depending on the node you might be running on, it can be necessary to adjust the default slurm scripts within `collect_metrics.py` and `scaling_analysis.py`. This can include: adding an account, adding a GPU request etc. If you have any trouble with this, contact your HPC admin team for help or search online. There are plenty of resources available that explain SLURM scripts.

---

## How to run everything

Tip: see all options with:

```bash
python collect_metrics.py --help
python scaling_analysis.py --help
```

From the repo root and within the virtual environment:

1. Edit `slurm_config.yaml` for your cluster (`CORES_PER_NODE`, `PARTITION`, `TIME_LIMIT`, `ACCOUNT`).

```bash
python collect_metrics.py --manual --submit # runs automated metric collection
```

```
# after jobs finish:
python collect_metrics.py --collect
python plot_metrics.py
```

This will:

1. Generate and submit a manual baseline job (`runs/manual/job.slurm`)
2. Generate and submit sweep jobs (`runs/run_*/job.slurm`) once manual job is finished
3. Collect logs into `runs/benchmark_summary.json` (with `--collect`)
4. Generate report at `performance_review.pdf`

---

## Outputs

* `runs/manual/` -> manual baseline run artifacts
* `runs/run_*/` -> sweep run artifacts (`params.json`, `lammps.log`, `run_result.json`)
* `runs/logs/*.log` -> consolidated log copies
* `runs/metrics_summary.json` -> parsed machine + run metrics
* `runs/performance_review.pdf` -> final benchmark report

Scaling (Slurm script generation):

* `runs/scaling_*/` -> generated scaling jobs (`job.slurm`, `params.json`)
* `runs/scaling_summary.json` -> machine + slurm config + list of generated scaling jobs
* `runs/scaling_*/submit_result.json` -> `sbatch` output (only when `--submit`)

---

## Scaling runs

1. Edit `slurm_config.yaml` for your cluster (`CORES_PER_NODE`, `PARTITION`, `TIME_LIMIT`, `ACCOUNT`).

2. Generate scripts (no submit):

```bash
python scaling_analysis.py
```

3. Generate + submit:

```bash
python scaling_analysis.py --submit
```

4. Generate PDF:
```
python plot_metrics.py
```

---

## Notes

* Default sweep dimensions are defined in `DEFAULT_SWEEP` in `collect_metrics.py`.
* The sweep input script consumes runtime variables (`ks`, `kacc`, `dcut`) via `-var`.
