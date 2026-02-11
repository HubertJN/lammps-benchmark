ls -ls mylammps/build/lmp*

mylammps/build/lmp -h | grep -i -E "KOKKOS|CUDA|OPENMP|AMPERE|86"

mylammps/build/lmp -h | grep -i -E "DIPOLE|KSPACE|ewald/dipole|lj/cut/dipole/long"