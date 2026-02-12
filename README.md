# LAMMPS (KOKKOS + CUDA + OpenMP) build for NVIDIA RTX A2000 with Python support

These instructions build LAMMPS from source with:
- KOKKOS acceleration
- CUDA GPU support
- OpenMP CPU threading
- NVIDIA RTX A2000 (Ampere, compute capability 8.6)
- Shared library + Python module install into a virtual environment
- (Optional, for dipole scripts) DIPOLE + KSPACE packages

## Prerequisites

- NVIDIA driver installed and working
- CUDA toolkit installed (compatible with your driver)
- A C++ compiler (e.g., gcc/g++)
- CMake
- Python 3 + venv

Verify CUDA is available:
```bash
nvidia-smi
nvcc --version
```

## Configure
```
cmake -S cmake -B build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_DIPOLE=ON \
  -D PKG_KSPACE=ON \
  -D CMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -D CMAKE_INSTALL_RPATH='$ORIGIN' \
  -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
```
## Build

```
cmake --build build -j [Number of Cores]
```

## Install the LAMMPS Python module into the active venv

```
cmake --build build --target install-python
```

## Verify build

```
bash verify.sh
```

## Run LAMMPS with basic script

```
mylammps/build/lmp -in in.FivesNoRescaleDipolarWCAFrancescoLangevinE.lmp
```