# kokkos-memory-pool [![CMake](https://github.com/matthew-mccall/kokkos-memory-pool/actions/workflows/cmake.yml/badge.svg)](https://github.com/matthew-mccall/kokkos-memory-pool/actions/workflows/cmake.yml)
A implementation of a memory pool in Kokkos.

### Build
1. Clone the repository.
2. Make sure to update the submodules with `git submodule init --update --recursive`.
3. Generate build files. For example, `cmake -B build .`
4. Build `cmake --build build`
5. Run the tests `cd build && ctest`